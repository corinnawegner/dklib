import types
import torch
from typing import Optional, Union, Set
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


class CustomUnmasker:
    def __init__(self, model_name: str, device: int = 0, dtype=torch.bfloat16):
        self._remote_code = True
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)

        self.model_name = model_name
    
    def __call__(self, text: str, max_new_tokens: int = 50):
        """
        Placeholder for generating or unmasking text.
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # For now just return the tokenized inputs (placeholder)
        return self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

def diffusion_generate_infilling(
    self,
    token_tensor: torch.LongTensor,
    attention_tensor: Optional[torch.LongTensor] = None,
    generation_config: Optional["DreamGenerationConfig"] = None,
    **kwargs,
):
    """
    Custom diffusion_generate that performs masked infilling.
    """
    generation_config = self._prepare_generation_config(generation_config, **kwargs)
    generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
    generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

    input_ids = token_tensor
    attention_mask = attention_tensor
    device = input_ids.device
    self._prepare_special_tokens(generation_config, device=device)

    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        input_ids_length=input_ids_length,
    )
    # we allow the max length to be exactly the input length, ignoring a valueerror and warning that this can lead to unexpected behaviour.
    #self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    pad_token_id = generation_config.pad_token_id

    # pad if needed
    if input_ids_length < max_length:
        pad_len = max_length - input_ids_length
        # compared to the original code, we pad with the pad token, not the mask token
        pad_token = torch.full((input_ids.size(0), pad_len), pad_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, pad_token], dim=-1)
        if attention_mask is not None:
            pad_mask = torch.ones((attention_mask.size(0), pad_len), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=-1)


    # skip expand — we want single completion
    result = self._sample(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        generation_tokens_hook_func=generation_tokens_hook_func,
        generation_logits_hook_func=generation_logits_hook_func,
    )

    return result

from typing import Iterable, Set
import torch
from transformers import PreTrainedTokenizerBase

def compute_banned_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    allow_numbers: bool = False,
    allowed_symbols: Optional[Set[str]] = None,
    ban_special_tokens: bool = True,
    extra_banned_strings: Optional[Set[str]] = None,
    allow_only_alpha: bool = True, # For the experiment with edgodicity breaking
) -> torch.LongTensor:
    """
    Scan tokenizer vocabulary and return token IDs that should be banned
    during generation (non-prose tokens + optionally special tokens + explicit strings).

    New options:
      - allow_only_alpha: when True, tokens that do not consist only of letters (after
        stripping common tokenization prefixes) are banned.
      - require_real_word: when True, tokens must also be recognized as real words via
        either 'wordfreq' (preferred) or the NLTK 'words' corpus. If no wordlist is
        available, this check is skipped with a warning.
    """

    if allowed_symbols is None:
        allowed_symbols = {
            ".", ",", "!", "?", "'", '"', ":", "-", "(", ")"
        }

    if extra_banned_strings is None:
        extra_banned_strings = {
            "ĊĊ",
            "Âł",
            "<br /><br />",
            "<|beginoftext|>",
            "Ġ",
            "\n",
            "\ ",
            "\r",
            ";"
        }

    banned_ids: Set[int] = set()
    vocab_size = len(tokenizer)

    # -----------------------
    # Vocabulary scan rules
    # -----------------------
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # Rule 2: numbers
        if not allow_numbers and any(c.isdigit() for c in token_str):
            banned_ids.add(token_id)
            continue

        normalized = token_str.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()
        # Rule 4: allow_only_alpha — ban tokens that are not pure letters after normalization
        if allow_only_alpha and (normalized == "" or not normalized.isalpha()):
            banned_ids.add(token_id)
            continue

    # -----------------------
    # Rule 6: special tokens
    # -----------------------
    if ban_special_tokens:
        banned_ids.update(tokenizer.all_special_ids)

    # -----------------------
    # Rule 7: explicit strings
    # -----------------------
    for s in extra_banned_strings:
        token_ids = tokenizer(
            s,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        for tid in token_ids:
            banned_ids.add(tid)

    banned_ids = torch.tensor(sorted(banned_ids), dtype=torch.long)

    print(f"Banned {len(banned_ids)} tokens (non-prose + special + explicit).")
    return banned_ids

def compute_first_token_banned_ids(tokenizer: PreTrainedTokenizerBase) -> torch.LongTensor:
    """Return token ids that SHOULD NOT be used as the very first token of a sentence.

    A token is disallowed if, after normalization, it is empty or its first character is
    not an uppercase ASCII letter (A-Z). This function is used to enforce that the
    first token starts with a capital letter.
    """

    banned: Set[int] = set()
    vocab_size = len(tokenizer)

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = token_str.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()
        if normalized == "":
            banned.add(token_id)
            continue
        first_char = normalized[0]
        # require an ASCII uppercase letter
        if not (first_char.isalpha() and first_char.isupper() and "A" <= first_char <= "Z"):
            banned.add(token_id)

    banned_ids = torch.tensor(sorted(banned), dtype=torch.long)
    print(f"First-token capitalization: banned {len(banned_ids)} tokens.")
    return banned_ids


def make_first_token_capitalized_logits_hook(banned_token_ids: torch.LongTensor):
    """Logits hook that bans the given token ids only at position 0 (first token)."""

    def logits_hook(step, x_t, logits):
        # logits: [batch, seq_len, vocab]
        if logits.size(1) > 0 and banned_token_ids.numel() > 0:
            logits[:, 0, banned_token_ids.to(logits.device)] = float("-inf")
        return logits

    return logits_hook

def make_ban_tokens_logits_hook(banned_token_ids: torch.LongTensor):
    """
    Returns a logits hook that bans specific token IDs during generation.
    """

    def logits_hook(step, x_t, logits):
        # logits: [batch, seq_len, vocab]
        logits[:, :, banned_token_ids.to(logits.device)] = float("-inf")
        return logits

    return logits_hook

def build_dream_substitutions(
    substitutions: torch.LongTensor,   # [1, num_masks, 4]
    final_tokens: torch.LongTensor,      # [1, seq_len] diffusion output
    history: list[torch.LongTensor],     # list of [1, seq_len], length = num mask
):
    """
    Takes information from the substitutions tensor about the masking positions for each uturn step. Extracts the final token id from final_tokens to the corresponding 
    masked token. Adds it to substitutions in the correct place. Also adds the unmasking step from the history (checks where in the history this token position was first not the mask token id)
    And adds the final token id after unmasking.
    Returns:
        substitutions: [num_uturns, num_masks, 4]
    """
    # print('starting sentence: ', sent_ind)
    #masked_token_sub_inds = torch.nonzero(
    #    (substitutions[:, 2] == -1) & (substitutions[:, 0] >= 0) #extract masked token positions by checking where we have final id = -1 and position not -1
    #)

    # Add unmasking step from history
    for sent_id in range(substitutions.shape[0]):
        for token_unmask in range(substitutions.shape[1]):
            #print("substitutions.shape:", substitutions.shape)
            tok_pos = substitutions[sent_id, token_unmask, 0]
            #print("tok_pos:", tok_pos)
            if tok_pos < 0:
                continue
            new_id = final_tokens[sent_id, tok_pos]
            history_token_pos = [tokens[sent_id, tok_pos] for tokens in history]
            step_at_unmasking = history_token_pos.index(new_id) # Where history_token_pos is not the mask token id for the first time
            substitutions[sent_id, token_unmask, 2] = new_id
            substitutions[sent_id, token_unmask, 3] = step_at_unmasking

    return substitutions

def unmask_batch_dream(
    masked_token_tensor: torch.LongTensor,          # [num_runs, seq_len]
    attention_tensor: torch.Tensor,                 # [num_runs, seq_len]
    substitutions_old: torch.LongTensor,           # [num_runs, max_masks, 4]
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    #mask_frac: float = 0.5,
):
    """
    Perform masked unmasking using Dream diffusion model within a FillMaskPipeline.
    Args:
        masked_token_tensor: [B: number of uturn steps, L: num tokens] tensor with masked tokens
        attention_tensor: [B, L] attention mask
        substitutions_old: [B, M: max number masks, 4] substitutions before unmasking, filled with masked positions and previous token ids at those positions
        pipeline: FillMaskPipeline with Dream model
        mask_frac: Fraction of tokens to mask (for diffusion steps)
    Returns:
        masked_token_tensor: Updated in-place with unmasked tokens
        substitutions_new: [B, M, 4] new substitutions after unmasking
    """
    tok = pipeline.tokenizer
    model = pipeline.model
    device = masked_token_tensor.device

    # In the dream case I have a pre-filled substitutions tensor with previous token ids and positions stored in the substitutions tensor. However, I don't need to select the
    # Unmasking positions myself, diffusion_generate_infilling does that. So I need to do the following here:
    # - Obtain the masked token tensor after unmasking
    # - Update the substitution tensor with the unmasking step! The final token will be done later anyway with the information from the masked_token_tensor

    # compute banned token IDs ONCE — only allow alphabetic tokens and prefer real English words
    if not hasattr(pipeline, "_banned_ids"):
        pipeline._banned_ids = compute_banned_token_ids(
            tok, allow_only_alpha=True
        )

    banned_ids = pipeline._banned_ids
    logits_hook = make_ban_tokens_logits_hook(banned_ids)

    # Optionally enforce that the first token starts with an uppercase ASCII letter.
    generation_logits_hook_func = logits_hook
    if getattr(pipeline, "_require_capitalized_start", True):
        if not hasattr(pipeline, "_first_token_banned_ids"):
            pipeline._first_token_banned_ids = compute_first_token_banned_ids(tok)
        first_banned = pipeline._first_token_banned_ids
        first_hook = make_first_token_capitalized_logits_hook(first_banned)

        def _combined_hook(step, x_t, logits):
            logits = logits_hook(step, x_t, logits)
            logits = first_hook(step, x_t, logits)
            return logits

        generation_logits_hook_func = _combined_hook


    # Bind custom diffusion method (TODO: Think where to put this to avoid repeated binding)
    model.diffusion_generate_infilling = types.MethodType(
        diffusion_generate_infilling, model
    )

    batch_size, seq_len = masked_token_tensor.shape # in the sequential case it is 1, seq_len

    # Safety: never pass negatives to Dream
    masked_token_tensor = masked_token_tensor.clone()
    masked_token_tensor[masked_token_tensor < 0] = tok.mask_token_id

    # --- run Dream diffusion ---
    output = model.diffusion_generate_infilling(
        token_tensor=masked_token_tensor,
        attention_mask=attention_tensor,
        max_length=seq_len,
        output_history=True,
        return_dict_in_generate=True,
        steps=masked_token_tensor.shape[1],#max(1, int(mask_frac * seq_len)),  # ensure at least 1 step
        temperature=1.0,
        top_p=0.95,
        alg="origin",
        alg_temp=0.0,
        generation_logits_hook_func=generation_logits_hook_func,
    )

    final_tokens = output.sequences[:, :seq_len].clone()

    # --- update substitutions, not in place like for bert ---
    substitutions_new = build_dream_substitutions(
        substitutions = substitutions_old, #original_tokens=original_tokens,
        final_tokens=final_tokens,
        history=output.history,
    )

    # Rewrite masked token tensor
    masked_token_tensor[:] = final_tokens

    return masked_token_tensor, substitutions_new
"""
def compute_banned_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    allow_numbers: bool = False,
    allow_newlines: bool = False,
    allowed_symbols: Optional[Set[str]] = None,
    ban_special_tokens: bool = True,
) -> torch.LongTensor:
"""
    #Scan tokenizer vocabulary and return token IDs that should be banned
    #during generation (non-prose tokens + optionally special tokens).

    ##Rules:
      #- Newlines banned by default
      #- Numbers banned by default
      #- Code / non-prose symbols banned
      #- Special tokens always banned (recommended)

    #Returns:
     #   torch.LongTensor of banned token IDs
"""

    if allowed_symbols is None:
        # Standard English punctuation we allow
        allowed_symbols = {
            ".", ",", "!", "?", "'", '"', ";", ":", "-", "(", ")", " "
        }

    banned_ids: Set[int] = set()
    vocab_size = len(tokenizer)

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # --- RULE 1: BAN NEWLINES ---
        if not allow_newlines and ("\n" in token_str or "\r" in token_str):
            banned_ids.add(token_id)
            continue

        # --- RULE 2: BAN NUMBERS ---
        if not allow_numbers and any(c.isdigit() for c in token_str):
            banned_ids.add(token_id)
            continue

        # --- RULE 3: BAN CODE / NON-PROSE SYMBOLS ---
        for char in token_str:
            if not char.isalpha() and char not in allowed_symbols:
                banned_ids.add(token_id)
                break

    # --- RULE 4: BAN SPECIAL TOKENS (CRITICAL) ---
    if ban_special_tokens:
        banned_ids.update(tokenizer.all_special_ids)

    banned_ids = torch.tensor(sorted(banned_ids), dtype=torch.long)

    print(f"Banned {len(banned_ids)} tokens (non-prose + special).")
    return banned_ids
"""
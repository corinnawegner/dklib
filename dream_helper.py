import types
import torch
from typing import Optional, Union, Set, Callable, Iterable
import math
import random
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from dream_model.modeling_dream import DreamModel
from dream_model.generation_utils import (
    DreamGenerationConfig,
    DreamModelOutput,
    sample_tokens
)
import logging
from .sentiment_steering import (
    _init_sentiment_model,
    _compute_sentiment_vector,
    _compute_sentiment_score,
    _validate_sentiment
)

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# --------------------------------------------------
# Custom Dream Model
# --------------------------------------------------
class CustomDreamModel(DreamModel):
    """
    Subclass of DreamModel allowing:
      - Binding custom diffusion_generate_infilling method
      - Token banning and grammar checking
    """
    def __init__(self, config, tokenizer=None, grammar_checker: Optional[Callable] = None):
        super().__init__(config)
        self.tokenizer = tokenizer  # optional tokenizer reference
        self.grammar_checker = grammar_checker
        self._banned_ids = None  # can be set later
        self._first_token_banned_ids = None
        
        # Bind diffusion_generate_infilling method to this model instance
        self.diffusion_generate_infilling = types.MethodType(
            _diffusion_generate_infilling_impl, self
        )

# ============================================================================
# GRAMMAR CHECKING
# ============================================================================

def _init_grammar_checker(method: str = "gpt"):
    """
    Initialize grammar checker based on method.
    Returns a callable that takes text and returns True if grammatically correct.
    """
    if method == "gpt":
        try:
            from openai import OpenAI
            api_key = __import__('os').environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not set. Grammar checking disabled.")
                return None
            
            client = OpenAI(api_key=api_key)
            
            def check_grammar_gpt(text: str, model: str = "gpt-4.1-nano") -> bool:
                """Check if text is grammatically correct using GPT."""
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict English grammar checker. "
                                    "Return YES if the text is grammatically correct. "
                                    "Return NO if the text has any grammar errors. "
                                    "Respond with exactly ONE token: YES or NO. No explanation."
                                )
                            },
                            {"role": "user", "content": text},
                        ],
                        max_tokens=1,
                        temperature=0,
                    )
                    result = response.choices[0].message.content.strip().upper()
                    return result == "YES"
                except Exception as e:
                    print(f"Warning: Grammar check failed: {e}")
                    return True  # Default to allow if check fails
            
            return check_grammar_gpt
        except ImportError:
            print("Warning: OpenAI client not available. Grammar checking disabled.")
            return None
    else:
        print(f"Warning: Grammar checking method '{method}' not implemented.")
        return None


class CustomUnmasker:
    def __init__(self, model_name: str, device: int = 0, sentiment_model: Optional[str] = None, dtype=torch.bfloat16, local_model_path: Optional[str] = None):
        self._remote_code = True
        self.device = device
        
        # Determine model path: use local path if provided, otherwise use model_name from HuggingFace
        if local_model_path is not None:
            # Load from local submodule
            import os
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"Local model path does not exist: {local_model_path}")
            model_path = local_model_path
            print(f"Loading Dream model from local path: {model_path}")
        else:
            # Load from HuggingFace Hub
            model_path = model_name
            print(f"Loading Dream model from HuggingFace Hub: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model using CustomDreamModel (diffusion_generate_infilling is bound in __init__)
        self.model = CustomDreamModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            tokenizer=self.tokenizer,
        ).to(device)

        self.model_name = model_name
        self.sentiment_model = sentiment_model
        
        # Initialize sentiment model (optional, for sentiment steering)
        if self.sentiment_model is not None:
            self.sentiment_tokenizer, self.sentiment_model = _init_sentiment_model(self.sentiment_model)
            self.sentiment_model = self.sentiment_model.to(device)
            
def _diffusion_generate_infilling_impl(
    self,
    token_tensor: torch.LongTensor,
    attention_tensor: Optional[torch.LongTensor] = None,
    generation_config: Optional["DreamGenerationConfig"] = None,
    **kwargs,
):
    """
    Custom diffusion_generate that performs masked infilling.
    This function is bound to CustomDreamModel instances in CustomDreamModel.__init__.
    """
    generation_config = self._prepare_generation_config(generation_config, **kwargs)

    # Apply banned token hook
    banned_token_ids = compute_banned_token_ids(self.tokenizer)
    #logger.debug(f"Banned token IDs computed: {banned_token_ids}")

    # Create logits hook
    generation_logits_hook_func = make_ban_tokens_logits_hook(banned_token_ids)

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

    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    pad_token_id = generation_config.pad_token_id

    # Pad if needed
    if input_ids_length < max_length:
        pad_len = max_length - input_ids_length
        pad_token = torch.full((input_ids.size(0), pad_len), pad_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, pad_token], dim=-1)
        if attention_mask is not None:
            pad_mask = torch.ones((attention_mask.size(0), pad_len), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=-1)

    # Skip expand — we want single completion
    result = self._sample(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        generation_logits_hook_func=generation_logits_hook_func,
        generation_tokens_hook_func = lambda step, x_t, logits: x_t # Pass the missing argument
    )

    return result

def compute_banned_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    allow_numbers: bool = False,
    allowed_symbols: Optional[Set[str]] = None,
    ban_special_tokens: bool = True,
    extra_banned_strings: Optional[Set[str]] = None,
    allow_only_alpha: bool = False,
    require_real_word: bool = False,
    strict_real_word: bool = False,  # New flag for stricter real-word enforcement
) -> torch.LongTensor:
    """
    Scan tokenizer vocabulary and return token IDs that should be banned
    during generation (non-prose tokens + optionally special tokens + explicit strings).

    Parameters:
      - strict_real_word: when True, bans tokens that are subwords or require multiple tokens to form a valid word.
    """

    if allowed_symbols is None:
        allowed_symbols = {
            ".", ",", "!", "?", "'", '"', ":", "-", "(", ")"
        }

    if extra_banned_strings is None:
        extra_banned_strings = {
            #' ',
            "\n",
            "\r",
            #":", "(", ")", "-", "/", "?", "!", # Very strict
            "<|endoftext|>",
            "ĊĊ", "•",
            "Âł",
            "<br>",
            "<br/>",
            "Ċ", "^"
            "Ã", "#", "*"
            "Ĺ", "â","Ģ","¢","Ė","Ī","Ļ","Ĳ","Ď","Ė","Ē",
              "Ŀ","Ń","Ņ","Ŋ","Ŕ","Ŗ","Ş","Ť","Ŧ","Ũ","Ū","Ŭ","Ů","Ű","Ų", "Ŵ","Ŷ","Ÿ","Ź","Ż","Ž", "Ġ", "[","]",
              "<", ">", "{","}","%","^","*","_","+","=","\\","|","~","`", ".\n", "?\n",",\n","!\n", ":\n", ",\n" ,";\n", ")\n"
        }

    banned_ids: Set[int] = set()
    vocab_size = len(tokenizer)

    # Centralized import handling for optional dependencies
    word_frequency = None
    nltk = None
    nltk_words = None

    # Initialize wordlist for real-word checking if needed
    word_list = None
    if require_real_word:
        if word_frequency is None and nltk_words is None:
            print("Warning: Real-word checking is disabled due to missing dependencies.")
            require_real_word = False
        elif word_frequency:
            # Use wordfreq for real-word checking
            pass
        elif nltk_words:
            # Use nltk for real-word checking
            pass

    # -----------------------
    # Vocabulary scan rules
    # -----------------------
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # Rule 2: numbers
        if not allow_numbers and any(c.isdigit() for c in token_str):
            banned_ids.add(token_id)
            continue

        # Rule 2b: ban tokens containing newlines or carriage returns
        if "\n" in token_str or "\r" in token_str:
            banned_ids.add(token_id)
            continue

        # Rule 3: explicit banned strings
        if any(banned in token_str for banned in extra_banned_strings):
            banned_ids.add(token_id)
            continue

        normalized = token_str.replace("\u0120", "").replace("\u2581", "").replace("\u010a", "").strip()

        # Rule 4: allow_only_alpha — ban tokens that are not pure letters after normalization
        if allow_only_alpha and (normalized == "" or not normalized.isalpha()):
            banned_ids.add(token_id)
            continue

        # Rule 5: require_real_word — ban single letters and non-dictionary words
        if require_real_word and normalized != "":
            # Ban single letters
            if len(normalized) == 1:
                banned_ids.add(token_id)
                continue
            # Check if it's a real word
            if word_list == 'wordfreq':
                if word_frequency:
                    freq = word_frequency(normalized.lower(), 'en')
                    if freq == 0:
                        banned_ids.add(token_id)
                        continue
                else:
                    print("Warning: wordfreq is not available.")
            elif isinstance(word_list, set):
                if normalized.lower() not in word_list:
                    banned_ids.add(token_id)
                    continue

        # Rule 6: strict_real_word — ban subwords or compound tokens
        if strict_real_word:
            if token_str.startswith("##") or " " in token_str:
                banned_ids.add(token_id)
                continue

    # -----------------------
    # Rule 7: special tokens
    # -----------------------
    if ban_special_tokens:
        banned_ids.update(tokenizer.all_special_ids)

    return torch.tensor(sorted(banned_ids), dtype=torch.long)

def compute_first_token_allowed_ids(tokenizer: PreTrainedTokenizerBase) -> torch.LongTensor:
    """
    Return token IDs that are allowed as the very first token of a sentence.
    A token is allowed if it starts with an uppercase ASCII letter (A-Z).
    """
    allowed: Set[int] = set()
    vocab_size = len(tokenizer)

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = token_str.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()
        if normalized and "A" <= normalized[0] <= "Z":
            allowed.add(token_id)

    allowed_ids = torch.tensor(sorted(list(allowed)), dtype=torch.long)
    print(f"First-token capitalization: allowed {len(allowed_ids)} tokens.")
    return allowed_ids


def make_first_token_capitalized_logits_hook(
    tokenizer: PreTrainedTokenizerBase,
    first_token_allowed_ids: torch.LongTensor
):
    """
    Returns a logits hook that enforces the first token must start with a capital letter.
    It does this by creating a mask that bans all tokens except those in first_token_allowed_ids.
    """
    vocab_size = len(tokenizer)
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed_mask[first_token_allowed_ids] = True

    def logits_hook(step, x_t, logits):
        # logits: [batch, seq_len, vocab]
        if logits.size(1) > 0:
            # Ban all tokens that are not in the allowed list for the first position
            logits[:, 0, ~allowed_mask.to(logits.device)] = float("-inf")
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


def _validate_grammar(
    final_tokens: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    grammar_checker: Callable,
    device: torch.device = torch.device("cpu"),
    banned_tokens_per_step: Optional[dict] = None,  # Dictionary to track banned tokens per step
) -> bool:
    """
    Validate grammatical correctness of generated tokens.

    Args:
        final_tokens: [B, L] tensor of token IDs
        tokenizer: Tokenizer for decoding
        grammar_checker: Callable that takes text and returns True if grammatically correct
        device: Device to use
        banned_tokens_per_step: Dictionary to track banned tokens for the current step

    Returns:
        bool: True if all sentences are grammatically correct, False otherwise
    """
    batch_size = final_tokens.shape[0]
    all_valid = True

    for batch_idx in range(batch_size):
        text = tokenizer.decode(final_tokens[batch_idx], skip_special_tokens=True)

        if not grammar_checker(text):
            print(f"  Sentence {batch_idx} failed grammar check.")
            
            # Identify the first unmasked token
            mask_token_id = tokenizer.mask_token_id
            first_unmasked_token_idx = (final_tokens[batch_idx] != mask_token_id).nonzero(as_tuple=True)[0][0]
            first_unmasked_token_id = final_tokens[batch_idx, first_unmasked_token_idx].item()

            # Exclude the token ID that caused the failure for the current step
            if banned_tokens_per_step is not None:
                if batch_idx not in banned_tokens_per_step:
                    banned_tokens_per_step[batch_idx] = set()
                banned_tokens_per_step[batch_idx].add(first_unmasked_token_id)

            # Set the token ID to -1 to prevent its generation in this step
            final_tokens[batch_idx, first_unmasked_token_idx] = -1
            all_valid = False
        else:
            print(f"  Sentence {batch_idx} passed grammar check. ✓")

    return all_valid

def unmask_batch_dream(
    masked_token_tensor: torch.LongTensor,         # [num_runs, seq_len]
    attention_tensor: torch.Tensor,                # [num_runs, seq_len]
    substitutions_old: torch.LongTensor,           # [num_runs, max_masks, 4]
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    illegal_tokens: Optional[torch.LongTensor] = None,  # Additional tokens to ban for this call
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

    # compute banned token IDs ONCE — read constraints from pipeline attributes
    if not hasattr(pipeline, "_banned_ids"):
        allow_alpha = getattr(pipeline, "_allow_only_alpha", False)
        allow_nums = getattr(pipeline, "_allow_numbers", False)
        require_words = getattr(pipeline, "_require_real_word", False)
        strict_words = getattr(pipeline, "_strict_real_word", False)  # Retrieve the strict_real_word flag
        pipeline._banned_ids = compute_banned_token_ids(
            tok,
            allow_only_alpha=allow_alpha,
            allow_numbers=allow_nums,
            require_real_word=require_words,
            strict_real_word=strict_words,  # Pass the flag to the function
        )

    banned_ids = pipeline._banned_ids
    
    # Merge with additional illegal tokens if provided
    if illegal_tokens is not None and illegal_tokens.numel() > 0:
        # Concatenate and get unique IDs
        banned_ids = torch.cat([banned_ids.to(illegal_tokens.device), illegal_tokens]).unique()
        print(f"  Banning {illegal_tokens.numel()} additional token(s) for this attempt.")
    
    logits_hook = make_ban_tokens_logits_hook(banned_ids)

    # Optionally enforce that the first token starts with an uppercase ASCII letter.
    generation_logits_hook_func = logits_hook
    #if getattr(pipeline, "_require_capitalized_start", True): ALWAYS ENFORCE
    if not hasattr(pipeline, "_first_token_allowed_ids"):
        pipeline._first_token_allowed_ids = compute_first_token_allowed_ids(tok)
    first_allowed = pipeline._first_token_allowed_ids
    first_hook = make_first_token_capitalized_logits_hook(tok, first_allowed)

    def _combined_hook(step, x_t, logits):
        logits = logits_hook(step, x_t, logits)
        logits = first_hook(step, x_t, logits)
        return logits

    generation_logits_hook_func = _combined_hook


    batch_size, seq_len = masked_token_tensor.shape # in the sequential case it is 1, seq_len

    # Safety: never pass negatives to Dream
    masked_token_tensor = masked_token_tensor.clone()
    masked_token_tensor[masked_token_tensor < 0] = tok.mask_token_id

    # --- run Dream diffusion ---
    # diffusion_generate_infilling is already bound to model in CustomUnmasker.__init__
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
        substitutions = substitutions_old,
        final_tokens=final_tokens,
        history=output.history,
    )

    # Rewrite masked token tensor
    masked_token_tensor[:] = final_tokens

    return masked_token_tensor, substitutions_new

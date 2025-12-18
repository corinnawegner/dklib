import types
import torch
from typing import Optional
import transformers
from transformers import AutoModel, AutoTokenizer

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
        generation_logits_hook_func=lambda step, x, logits: logits,  # ✅ just a lambda
    )

    return result

def unmask_batch_dream(
    masked_token_tensor: torch.LongTensor,
    attention_tensor: torch.Tensor,
    substitutions_old: torch.LongTensor,
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    mask_frac: float = 0.5,
):
    """
    Dream unmasking with BERT-compatible substitutions.
    """
    model = pipeline.model
    tok = pipeline.tokenizer
    device = masked_token_tensor.device

    # Bind custom diffusion method
    model.diffusion_generate_infilling = types.MethodType(
        diffusion_generate_infilling, model
    )

    # --- reconstruct ORIGINAL tokens (before masking) ---
    # substitutions_old[:, :, 1] stores true originals
    original_tokens = masked_token_tensor.clone()
    for b in range(substitutions_old.shape[0]):
        mask = substitutions_old[b, :, 0] >= 0
        pos = substitutions_old[b, mask, 0]
        vals = substitutions_old[b, mask, 1]
        original_tokens[b, pos] = vals

    # Safety: never pass negatives to Dream
    masked_token_tensor = masked_token_tensor.clone()
    masked_token_tensor[masked_token_tensor < 0] = tok.mask_token_id

    # --- run Dream diffusion ---
    output = model.diffusion_generate_infilling(
        token_tensor=masked_token_tensor,
        attention_mask=attention_tensor,
        max_length=masked_token_tensor.shape[1],
        output_history=True,
        return_dict_in_generate=True,
        steps=512, #int(mask_frac * masked_token_tensor.shape[1]),
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )

    final_tokens = output.sequences[:, :masked_token_tensor.shape[1]].clone()

    # --- build BERT-compatible substitutions ---
    substitutions_new = build_dream_substitutions(
        original_tokens=original_tokens,
        masked_tokens=masked_token_tensor,
        final_tokens=final_tokens,
        history=output.history,
        mask_token_id=tok.mask_token_id,
        device=device,
    )

    # Update token tensor in place
    masked_token_tensor[:] = final_tokens

    return masked_token_tensor, substitutions_new




def fill_unmask_steps_from_history(output, original_tokens, device):
    """
    Build a unified [batch_size, max_changes, 4] substitutions tensor.
    Each row has: [token_position, token_before, token_after, diffusion_step]
    """
    histories = output.history  # list of [B, seq_len]
    num_steps = len(histories)
    batch_size = histories[0].shape[0]
    seq_len = histories[0].shape[1]

    history_stack = torch.stack(histories, dim=0).to(device)

    # Collect per-sentence changes
    all_changes = []
    max_changes = 0

    for b in range(batch_size):
        changes = []
        for t in range(1, num_steps):
            prev_tokens = history_stack[t - 1, b]
            curr_tokens = history_stack[t, b]

            diff_mask = prev_tokens != curr_tokens
            changed_positions = torch.nonzero(diff_mask, as_tuple=False).squeeze(-1)

            for pos in changed_positions:
                changes.append([
                    pos.item(),                 # token position
                    prev_tokens[pos].item(),    # token before
                    curr_tokens[pos].item(),    # token after
                    t - 1                       # diffusion step
                ])

        all_changes.append(changes)
        max_changes = max(max_changes, len(changes))

    # Create padded tensor: [batch_size, max_changes, 4]
    substitutions = torch.full(
        (batch_size, max_changes, 4), -1, dtype=torch.int64, device=device
    )

    for b, changes in enumerate(all_changes):
        if len(changes) > 0:
            substitutions[b, :len(changes)] = torch.tensor(changes, dtype=torch.int64, device=device)

    return substitutions

from typing import Iterable, Set
import torch
from transformers import PreTrainedTokenizerBase


def fill_unmask_steps_from_history(output, original_tokens, device, mask_token_id=None):
    """
    Build a unified [batch_size, max_changes, 4] substitutions tensor from Dream diffusion history,
    in the same format as BERT-style unmask_batch.

    Args:
        output: Diffusion generation output, must have `history` (list of [B, seq_len]).
        original_tokens: The original masked token tensor [B, seq_len].
        device: torch.device
        mask_token_id: Optional mask token ID, used to initialize slot 2 for yet-to-be-unmasked tokens.

    Returns:
        substitutions: [B, max_changes, 4] tensor
    """
    histories = output.history  # list of [B, seq_len]
    num_steps = len(histories)
    batch_size, seq_len = histories[0].shape

    history_stack = torch.stack(histories, dim=0).to(device)

    all_changes = []
    max_changes = 0

    for b in range(batch_size):
        changes = []
        for t in range(1, num_steps):
            prev_tokens = history_stack[t - 1, b]
            curr_tokens = history_stack[t, b]

            diff_mask = prev_tokens != curr_tokens
            changed_positions = torch.nonzero(diff_mask, as_tuple=False).squeeze(-1)

            for pos in changed_positions:
                changes.append([
                    pos.item(),                 # token position
                    prev_tokens[pos].item(),    # token before
                    curr_tokens[pos].item(),    # token after
                    t - 1                       # diffusion step
                ])
        all_changes.append(changes)
        max_changes = max(max_changes, len(changes))

    # Initialize substitutions tensor
    substitutions = torch.full(
        (batch_size, max_changes, 4),
        -1,
        dtype=torch.int64,
        device=device
    )

    for b, changes in enumerate(all_changes):
        if len(changes) > 0:
            # Fill the positions for tokens that actually changed
            substitutions[b, :len(changes), :] = torch.tensor(changes, dtype=torch.int64, device=device)

    # For slots that are still -1 (no diffusion yet), initialize slot 2 to mask_token_id
    if mask_token_id is None:
        mask_token_id = original_tokens.new_full((1,), -1).item()
    mask_positions = substitutions[:, :, 2] == -1
    substitutions[:, :, 2][mask_positions] = mask_token_id

    return substitutions


def compute_banned_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    allow_numbers: bool = False,
    allow_newlines: bool = False,
    allowed_symbols: Set[str] | None = None,
) -> torch.LongTensor:
    """
    Scan tokenizer vocabulary and return token IDs that should be banned
    (non-prose tokens: numbers, code symbols, newlines).

    Returns:
        torch.LongTensor of banned token IDs
    """

    if allowed_symbols is None:
        # Standard English punctuation we allow
        allowed_symbols = {
            ".", ",", "!", "?", "'", '"', ";", ":", "-", "(", ")", " "
        }

    banned_ids = []
    vocab_size = len(tokenizer)

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # --- RULE 1: BAN NEWLINES ---
        if not allow_newlines and ("\n" in token_str or "\r" in token_str):
            banned_ids.append(token_id)
            continue

        # --- RULE 2: BAN NUMBERS ---
        if not allow_numbers and any(c.isdigit() for c in token_str):
            banned_ids.append(token_id)
            continue

        # --- RULE 3: BAN CODE-LIKE SYMBOLS ---
        for char in token_str:
            if not char.isalpha() and char not in allowed_symbols:
                banned_ids.append(token_id)
                break

    banned_ids = torch.tensor(banned_ids, dtype=torch.long)

    print(f"Banned {len(banned_ids)} non-prose tokens.")
    return banned_ids

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
    *,
    original_tokens: torch.LongTensor,   # [B, L] BEFORE masking
    masked_tokens: torch.LongTensor,     # [B, L] AFTER masking
    final_tokens: torch.LongTensor,      # [B, L] diffusion output
    history: list[torch.LongTensor],     # list of [B, L]
    mask_token_id: int,
    device: torch.device,
):
    """
    Produce a BERT-compatible substitutions tensor from Dream diffusion.

    Returns:
        substitutions: [B, M, 4]
    """
    B, L = masked_tokens.shape
    num_steps = len(history)

    # Identify originally masked positions
    masked_pos = masked_tokens == mask_token_id          # [B, L]
    M = masked_pos.sum(dim=1).max().item()                # max masks per batch

    substitutions = torch.full(
        (B, M, 4),
        -1,
        dtype=torch.long,
        device=device
    )

    history_stack = torch.stack(history, dim=0)           # [T, B, L]

    for b in range(B):
        positions = torch.nonzero(masked_pos[b], as_tuple=False).squeeze(-1)

        for j, pos in enumerate(positions):
            # original token (pre-mask)
            orig_token = original_tokens[b, pos]

            # find FIRST diffusion step where this token changed
            step = -1
            for t in range(1, num_steps):
                if history_stack[t, b, pos] != history_stack[t - 1, b, pos]:
                    step = t - 1
                    break

            substitutions[b, j, 0] = pos
            substitutions[b, j, 1] = orig_token
            substitutions[b, j, 2] = final_tokens[b, pos]
            substitutions[b, j, 3] = step

    return substitutions

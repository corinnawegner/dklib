import types
import torch
from typing import Optional, Union, Set, Callable
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dream_model.generation_utils import (
    DreamGenerationConfig,
    DreamModelOutput,
    sample_tokens
)
from dream_model.modeling_dream import DreamModel
from dklib.banned_tokens import compute_banned_token_ids
from dklib.llm_loading import load_llm

# --------------------------------------------------
# Custom Dream Model
# --------------------------------------------------
class CustomDreamModel(DreamModel):
    """
    Subclass of DreamModel allowing:
      - Overriding _sample
      - Optional custom diffusion_generate_infilling
      - Token banning and grammar checking
    """
    def __init__(self, config, tokenizer=None, grammar_checker: Optional[Callable] = None):
        super().__init__(config)
        self.tokenizer = tokenizer  # optional tokenizer reference
        self.grammar_checker = grammar_checker
        self._banned_ids = None  # can be set later
        self._first_token_banned_ids = None

    # -------------------------------
    # Example: override _sample
    # -------------------------------
    @torch.no_grad()
    def _sample(self, input_ids, attention_mask, generation_config, generation_tokens_hook_func, generation_logits_hook_func):
        """
        Custom sampling logic here. Called by diffusion_generate internally.
        """
        print("CustomDreamModel._sample called!")
        # Example: just call the original sample logic
        x = input_ids.clone()

        # Default Dream behavior (simplified):
        # for each timestep, sample mask tokens
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        # pad to max_length
        max_length = generation_config.max_length
        x = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask).logits
            logits = logits[:, :-1]  # align logits for masking

            logits = generation_logits_hook_func(i, x, logits)
            mask_logits = logits[mask_index]

            # sample tokens
            _, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[mask_index] = x0

            x = generation_tokens_hook_func(i, x, logits)

        return DreamModelOutput(sequences=x, history=None)

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
    def __init__(self, model_name: str, device: int = 0, dtype=torch.bfloat16, local_model_path: Optional[str] = None, sentiment_model: Optional[str] = None, perplexity_model: Optional[str] = None, alpha_perplexity: float = 1.0):
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
        
        # Load model
        self.model = load_llm(
            model_path,
            device=device,
            eval_mode=False,
            torch_dtype=dtype,
        )

        self.model_name = model_name
        
        # Bind diffusion_generate_infilling function to the model
        self.model.diffusion_generate_infilling = types.MethodType(
            diffusion_generate_infilling, self.model
        )

    
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
    This function is bound to the Dream model in CustomUnmasker.__init__.
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


def _validate_and_resample_grammar(
    final_tokens: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    grammar_checker: Callable,
    max_retries: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.LongTensor:
    """
    Validate grammatical correctness of generated tokens and resample if needed.
    
    Args:
        final_tokens: [B, L] tensor of token IDs
        tokenizer: Tokenizer for decoding
        grammar_checker: Callable that takes text and returns True if grammatically correct
        max_retries: Max number of resampling attempts per sentence
        device: Device to use
    
    Returns:
        final_tokens: [B, L] tensor with grammatically-validated tokens
    """
    final_tokens = final_tokens.clone()
    batch_size = final_tokens.shape[0]
    
    for batch_idx in range(batch_size):
        # Decode the current sentence
        text = tokenizer.decode(final_tokens[batch_idx], skip_special_tokens=True)
        
        # Check if grammatically correct
        if not grammar_checker(text):
            print(f"  Sentence {batch_idx} failed grammar check. Resampling...")
            
            # Try resampling by randomly replacing tokens until grammar passes
            for retry in range(max_retries):
                # Clone the tokens for this attempt
                test_tokens = final_tokens[batch_idx].clone()
                
                # Randomly pick a non-special token position to resample
                special_ids = set(tokenizer.all_special_ids)
                valid_positions = [
                    i for i in range(len(test_tokens))
                    if test_tokens[i].item() not in special_ids
                ]
                
                if not valid_positions:
                    print(f"Retry {retry + 1}/{max_retries}: No valid positions to resample.")
                    continue
                
                # Pick a random position
                pos = valid_positions[torch.randint(0, len(valid_positions), (1,)).item()]
                
                # Resample a random token (excluding special tokens)
                vocab_size = len(tokenizer)
                while True:
                    new_token_id = torch.randint(0, vocab_size, (1,)).item()
                    if new_token_id not in special_ids:
                        break
                
                test_tokens[pos] = new_token_id
                test_text = tokenizer.decode(test_tokens, skip_special_tokens=True)
                
                if grammar_checker(test_text):
                    print(f"    Retry {retry + 1}/{max_retries}: Grammar passed! ✓")
                    final_tokens[batch_idx] = test_tokens
                    break
                else:
                    print(f"    Retry {retry + 1}/{max_retries}: Still incorrect, trying again...")
            else:
                # All retries exhausted
                print(f"  Could not fix grammar after {max_retries} retries. Using original tokens.")
        else:
            print(f"  Sentence {batch_idx} passed grammar check. ✓")
    
    return final_tokens


def unmask_batch_dream(
    masked_token_tensor: torch.LongTensor,         # [num_runs, seq_len]
    attention_tensor: torch.Tensor,                # [num_runs, seq_len]
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

    # compute banned token IDs ONCE — read constraints from pipeline attributes
    if not hasattr(pipeline, "_banned_ids"):
        pipeline._banned_ids = compute_banned_token_ids(
            tok,
            ban_numbers=getattr(pipeline, "_ban_numbers", True),
            ban_symbols=getattr(pipeline, "_ban_symbols", True),
            ban_unicode_artifacts=getattr(pipeline, "_ban_unicode_artifacts", True),
            ban_special_tokens=getattr(pipeline, "_ban_special_tokens", True),
            ban_non_alpha=getattr(pipeline, "_ban_non_alpha", False),
            ban_repeated_punctuation=getattr(pipeline, "_ban_repeated_punctuation", False),
            ban_crosslingual=getattr(pipeline, "_ban_crosslingual", False),
            require_real_word=getattr(pipeline, "_require_real_word", False),
            strict_real_word=getattr(pipeline, "_strict_real_word", False),
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
        substitutions = substitutions_old, #original_tokens=original_tokens,
        final_tokens=final_tokens,
        history=output.history,
    )

    # --- optional grammar validation with resampling ---
    if getattr(pipeline, "_validate_grammar", False):
        if not hasattr(pipeline, "_grammar_checker"):
            grammar_method = getattr(pipeline, "_grammar_method", "gpt")
            pipeline._grammar_checker = _init_grammar_checker(grammar_method)
        
        grammar_checker = pipeline._grammar_checker
        max_retries = getattr(pipeline, "_grammar_max_retries", 3)
        
        if grammar_checker is not None:
            final_tokens = _validate_and_resample_grammar(
                final_tokens,
                tok,
                grammar_checker,
                max_retries=max_retries,
                device=device,
            )
            # Rebuild substitutions with validated tokens
            substitutions_new = build_dream_substitutions(
                substitutions = substitutions_old,
                final_tokens=final_tokens,
                history=output.history,
            )

    # Rewrite masked token tensor
    masked_token_tensor[:] = final_tokens

    return masked_token_tensor, substitutions_new


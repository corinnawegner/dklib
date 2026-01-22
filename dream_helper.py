import types
import torch
from typing import Optional, Union, Set, Callable
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from dream_model.modeling_dream import DreamModel
from dream_model.generation_utils import (
    DreamGenerationConfig,
    DreamModelOutput,
    sample_tokens
)

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


# ============================================================================
# SENTIMENT STEERING
# ============================================================================

def _init_sentiment_model(sentiment_model_name: str = "SamLowe/roberta-base-go_emotions"):
    """
    Initialize sentiment/emotion model for steering.
    Returns (tokenizer, model) or (None, None) if unavailable.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        model.eval()
        
        print(f"Loaded sentiment model: {sentiment_model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"Warning: Failed to load sentiment model: {e}")
        return None, None


def _compute_sentiment_vector(
    text: str,
    sentiment_tokenizer,
    sentiment_model,
    device: torch.device = torch.device("cpu"),
):
    """
    Compute emotion/sentiment vector for text.
    Returns numpy array of emotion probabilities.
    """
    import torch.nn.functional as F
    
    inputs = sentiment_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs


def _sentiment_distance_to_neutral(sentiment_vector):
    """
    Compute distance of sentiment vector to neutral (all zeros).
    Returns L2 norm of the vector.
    """
    import numpy as np
    return np.linalg.norm(sentiment_vector)


def _get_emotion_index(emotion_name: str, sentiment_model) -> int:
    """
    Get the index of an emotion in the model's label set.
    
    Args:
        emotion_name: Name of the emotion (e.g., 'neutral', 'joy', 'sadness')
        sentiment_model: The sentiment/emotion model
    
    Returns:
        int: Index of the emotion in the model's config
    """
    if hasattr(sentiment_model, 'config') and hasattr(sentiment_model.config, 'id2label'):
        id2label = sentiment_model.config.id2label
        label2id = {v.lower(): k for k, v in id2label.items()}
        emotion_lower = emotion_name.lower()
        if emotion_lower in label2id:
            return label2id[emotion_lower]
        else:
            available = list(label2id.keys())
            raise ValueError(f"Emotion '{emotion_name}' not found. Available emotions: {available}")
    else:
        raise ValueError("Sentiment model does not have config.id2label attribute")


def _sentiment_distance_to_target(sentiment_vector, target_emotion_index: int) -> float:
    """
    Compute distance of sentiment vector to a target emotion.
    Only considers the probability of the target emotion.

    Args:
        sentiment_vector: Emotion probability distribution [num_emotions]
        target_emotion_index: Index of the target emotion

    Returns:
        float: Distance as 1 - probability of the target emotion
    """
    # Distance is defined as 1 minus the probability of the target emotion
    return sentiment_vector[target_emotion_index]


class CustomUnmasker:
    def __init__(self, model_name: str, device: int = 0, dtype=torch.bfloat16, local_model_path: Optional[str] = None):
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
        
        # Initialize sentiment model (optional, for sentiment steering)
        self.sentiment_tokenizer, self.sentiment_model = _init_sentiment_model()
        if self.sentiment_model is not None:
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
            "\n",
            "\r",
            "\ ",
            "<|endoftext|>",
            "ĊĊ",
            "Âł"
        }

    banned_ids: Set[int] = set()
    vocab_size = len(tokenizer)

    # Initialize wordlist for real-word checking if needed
    word_list = None
    if require_real_word:
        try:
            from wordfreq import word_frequency
            word_list = 'wordfreq'
        except ImportError:
            try:
                import nltk
                nltk.download('words', quiet=True)
                from nltk.corpus import words as nltk_words
                word_list = set(w.lower() for w in nltk_words.words())
            except (ImportError, LookupError):
                print("Warning: require_real_word=True but neither 'wordfreq' nor 'nltk' words corpus available. Skipping real-word check.")
                require_real_word = False

    # -----------------------
    # Vocabulary scan rules
    # -----------------------
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # Rule 2: numbers
        if not allow_numbers and any(c.isdigit() for c in token_str):
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
                from wordfreq import word_frequency
                freq = word_frequency(normalized.lower(), 'en')
                if freq == 0:
                    banned_ids.add(token_id)
                    continue
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


def _validate_sentiment(
    final_tokens: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    sentiment_tokenizer,
    sentiment_model,
    device: torch.device = torch.device("cpu"),
    sentiment_target: str = "neutral",
    previous_best_score: Optional[float] = None,
) -> tuple[bool, float]:
    """
    Validate that unmasked text is close to target sentiment/emotion.
    
    Args:
        final_tokens: [B, L] tensor of token IDs
        tokenizer: Dream tokenizer
        sentiment_tokenizer: Sentiment model tokenizer
        sentiment_model: Sentiment model
        device: Device to use
        sentiment_target: Target emotion (e.g., 'neutral', 'joy', 'sadness'). Default is 'neutral'.
        previous_best_score: If provided, only accept if current score is BETTER (lower distance).
                            If None, accept first attempt and return its score.
    
    Returns:
        tuple[bool, float]: (validation_passed, score) where score is the distance to target emotion.
                           validation_passed=True if this is first attempt OR score improved over previous_best_score.
    """
    batch_size = final_tokens.shape[0]
    
    # Get target emotion index
    try:
        target_idx = _get_emotion_index(sentiment_target, sentiment_model)
    except ValueError as e:
        print(f"Error: {e}")
        return False, float('inf')
    
    # Compute sentiment for all sentences in batch and take max distance (worst case)
    max_distance = 0.0
    for batch_idx in range(batch_size):
        text = tokenizer.decode(final_tokens[batch_idx], skip_special_tokens=True)
        sentiment_vector = _compute_sentiment_vector(
            text, sentiment_tokenizer, sentiment_model, device=device
        )
        distance = _sentiment_distance_to_target(sentiment_vector, target_idx)
        max_distance = max(max_distance, distance)
    
    # Determine validation result
    if previous_best_score is None:
        # First attempt: always accept and return score
        print(f"  First sentiment check - storing baseline score: {max_distance:.4f}")
        return True, max_distance
    else:
        # Subsequent attempts: only accept if improved (lower distance)
        improved = max_distance < previous_best_score
        if improved:
            print(f"  Sentiment improved: {max_distance:.4f} < {previous_best_score:.4f}. ✓")
        else:
            print(f"  Sentiment did not improve: {max_distance:.4f} >= {previous_best_score:.4f}. Retrying...")
        return improved, max_distance


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
        substitutions = substitutions_old,
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
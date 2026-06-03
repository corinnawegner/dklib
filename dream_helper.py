"""
Masked-diffusion unmasking helper.

Originally written for the Dream-org/Dream diffusion model. On the
``corinna_llaada`` branch this has been ported to the LLaDA-MoE diffusion LM
(``inclusionAI/LLaDA-MoE-7B-A1B-Base``). Public symbol names
(``CustomUnmasker``, ``unmask_batch_dream``, ``build_dream_substitutions``)
are kept for backward compatibility with existing call sites and notebooks.
"""
import types
import torch
import torch.nn.functional as F
from typing import Optional, Union, Set, Callable, Iterable
import math
import random
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
import logging
from .sentiment_steering import (
    _init_sentiment_model,
    _compute_sentiment_vector,
    _compute_sentiment_score,
    _validate_sentiment,
    _init_perplexity_model,
    _compute_perplexity,
    _validate_perplexity,
)

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# --------------------------------------------------
# LLaDA-MoE constants
# --------------------------------------------------
# Mask token id used by the LLaDA-MoE-7B-A1B tokenizer (see model card).
# The tokenizer does not always advertise ``mask_token_id`` so we fall back
# to this value if needed.
_LLADA_MOE_MASK_ID = 156895
_LLADA_MOE_MASK_STRING = "<|mask|>"  # display string only; not relied on by infill loop


class _InfillResult:
    """Lightweight stand-in for the Dream `DreamModelOutput`/HF GenerateOutput.

    Exposes ``.sequences`` (B, L) and optionally ``.history`` (list of B, L
    tensors) so that downstream code in :func:`unmask_batch_dream` and
    :func:`build_dream_substitutions` keeps working unchanged.
    """
    def __init__(self, sequences: torch.Tensor, history: Optional[list] = None):
        self.sequences = sequences
        self.history = history if history is not None else []


def _ensure_mask_token(tokenizer: PreTrainedTokenizerBase) -> int:
    """Make sure ``tokenizer.mask_token_id`` is set for LLaDA-MoE.

    The LLaDA-MoE tokenizer does not register a HuggingFace-style mask token
    out of the box. We patch it in so that the rest of the pipeline (which
    consistently uses ``tokenizer.mask_token_id``) works without changes.
    Returns the resolved mask token id.
    """
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is None:
        # Try to look up the default LLaDA-MoE mask id in the vocabulary.
        mid = _LLADA_MOE_MASK_ID
        try:
            tokenizer.mask_token = tokenizer.convert_ids_to_tokens(mid)
        except Exception:
            tokenizer.mask_token = _LLADA_MOE_MASK_STRING
        # Setting mask_token recomputes mask_token_id but only if the string
        # exists in the vocab; force it explicitly.
        try:
            tokenizer.mask_token_id = mid
        except Exception:
            pass
    return int(tokenizer.mask_token_id)


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
    def __init__(self, model_name: str, device: int = 0, sentiment_model: Optional[str] = None, dtype=torch.bfloat16, local_model_path: Optional[str] = None, perplexity_model: Optional[str] = None, alpha_perplexity: float = 1.0):
        self._remote_code = True
        self.device = device
        
        # Determine model path: use local path if provided, otherwise use model_name from HuggingFace
        if local_model_path is not None:
            # Load from local directory
            import os
            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"Local model path does not exist: {local_model_path}")
            model_path = local_model_path
            print(f"Loading LLaDA-MoE model from local path: {model_path}")
        else:
            # Load from HuggingFace Hub
            model_path = model_name
            print(f"Loading LLaDA-MoE model from HuggingFace Hub: {model_path}")
        
        # Load tokenizer (LLaDA-MoE requires trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _ensure_mask_token(self.tokenizer)

        # Load model. LLaDA-MoE is a masked-diffusion LM exposed via AutoModel
        # with custom remote code.
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        # Attach tokenizer + bind the masked-infill diffusion loop directly
        # onto the model instance so we can call
        # ``model.diffusion_generate_infilling(...)`` like the original Dream
        # code did.
        self.model.tokenizer = self.tokenizer
        self.model.diffusion_generate_infilling = types.MethodType(
            _diffusion_generate_infilling_impl, self.model
        )

        self.model_name = model_name
        self.sentiment_model_name = sentiment_model
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        
        # Initialize sentiment model (optional, for sentiment steering)
        if self.sentiment_model_name is not None:
            self.sentiment_tokenizer, self.sentiment_model = _init_sentiment_model(
                self.sentiment_model_name, sample_sentiment=True
            )
            if self.sentiment_model is None:
                raise RuntimeError(
                    f"Sentiment model '{self.sentiment_model_name}' failed to load. "
                    f"Pre-cache it on a login node with: "
                    f"python -c \"from transformers import AutoTokenizer, AutoModelForSequenceClassification; "
                    f"AutoTokenizer.from_pretrained('{self.sentiment_model_name}'); "
                    f"AutoModelForSequenceClassification.from_pretrained('{self.sentiment_model_name}')\""
                )
            self.sentiment_model = self.sentiment_model.to(device)

        # Initialize perplexity model (optional, for perplexity-guided steering)
        self.perplexity_model_name = perplexity_model
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        self._alpha_perplexity = alpha_perplexity

        if self.perplexity_model_name is not None:
            self._validate_perplexity = True
            self.perplexity_tokenizer, self.perplexity_model = _init_perplexity_model(
                self.perplexity_model_name, validate_perplexity=True
            )
            if self.perplexity_model is None:
                raise RuntimeError(
                    f"Perplexity model '{self.perplexity_model_name}' failed to load."
                )
            self.perplexity_model = self.perplexity_model.to(device)
        else:
            self._validate_perplexity = False


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-noise reparameterisation used by LLaDA's reference sampler."""
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute the number of mask positions to transfer over `steps` steps.

    Mirrors ``get_num_transfer_tokens`` from the LLaDA-MoE README.
    Returns a tensor of shape (B, steps) with non-negative integers that sum
    (per row) to the total number of masked positions in that row.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def _diffusion_generate_infilling_impl(
    self,
    token_tensor: torch.LongTensor,
    attention_tensor: Optional[torch.LongTensor] = None,
    generation_config=None,  # accepted for API compat; unused for LLaDA
    **kwargs,
):
    """Masked-infilling diffusion generation for LLaDA-MoE.

    Bound to the loaded model instance in :class:`CustomUnmasker`. Mirrors the
    public surface previously offered by the Dream model so that
    :func:`unmask_batch_dream` (and the rest of the pipeline) does not need to
    change.

    Recognised ``kwargs``:
        * ``steps`` (int, optional): number of diffusion steps. Defaults to
          the number of masked tokens (= one new token per step).
        * ``temperature`` (float, default 0.0): Gumbel-noise temperature.
        * ``generation_logits_hook_func`` (callable, optional): called as
          ``hook(step, x_t, logits)`` and may modify the logits in-place.
        * ``output_history`` (bool, default False): record per-step state.
        * ``return_dict_in_generate`` (bool, default False): kept for API
          compatibility (we always return a result object).

    Dream-specific kwargs (``top_p``, ``alg``, ``alg_temp``, ``max_length``,
    ``pad_token_id``) are accepted but ignored.
    """
    steps_arg = kwargs.get("steps", None)
    temperature = float(kwargs.get("temperature", 0.0))
    output_history = bool(kwargs.get("output_history", False))
    logits_hook = kwargs.get("generation_logits_hook_func", None)

    # Callers (e.g. unmask_batch_dream) pass the attention mask as
    # ``attention_mask=`` (the canonical HuggingFace kwarg name) rather than
    # the positional ``attention_tensor`` parameter, so it ends up in **kwargs.
    # Pick it up here so the model forward pass actually sees it.
    if attention_tensor is None:
        attention_tensor = kwargs.get("attention_mask", None)

    if logits_hook is None:
        # Default hook: ban non-prose tokens (matches the original Dream
        # fallback behaviour).
        banned_token_ids = compute_banned_token_ids(self.tokenizer)
        logits_hook = make_ban_tokens_logits_hook(banned_token_ids)

    mask_id = _ensure_mask_token(self.tokenizer)
    device = token_tensor.device

    x = token_tensor.clone()
    # Replace any negative sentinel values with the mask id (defensive: the
    # caller in unmask_batch_dream already does this, but keep it here too).
    x[x < 0] = mask_id

    initial_mask_index = (x == mask_id)
    n_masked = int(initial_mask_index.sum(dim=1).max().item())

    if steps_arg is None:
        steps = max(n_masked, 1)
    else:
        steps = max(int(steps_arg), 1)

    # Schedule of how many tokens to commit per step (per batch row).
    num_transfer_tokens = _get_num_transfer_tokens(initial_mask_index, steps).to(device)

    history = [x.clone()] if output_history else None

    for step in range(steps):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # Forward pass through the LLaDA-MoE model. AutoModel returns an
        # object with a ``.logits`` attribute.
        outputs = self(x, attention_mask=attention_tensor)
        logits = outputs.logits

        # Apply caller-provided logits constraints (token bans, capitalised
        # first token, per-position bans, ...).
        logits = logits_hook(step, x, logits)

        # Sample candidate tokens for every position.
        if temperature == 0:
            x0 = torch.argmax(logits, dim=-1)
        else:
            noisy = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(noisy, dim=-1)

        # Confidence = softmax probability of the chosen token, computed on
        # the (already constrained) logits so that banned tokens have zero
        # probability and never get selected for transfer.
        probs = F.softmax(logits.to(torch.float32), dim=-1)
        x0_conf = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        # Only consider currently-masked positions; keep prompt tokens fixed.
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(
            mask_index,
            x0_conf,
            torch.full_like(x0_conf, float("-inf")),
        )

        # Transfer the top-k highest-confidence positions per row.
        transfer_index = torch.zeros_like(x, dtype=torch.bool)
        for b in range(x.size(0)):
            k = int(num_transfer_tokens[b, step].item())
            if k <= 0:
                continue
            # Cap k at the number of remaining masked positions for safety.
            k = min(k, int(mask_index[b].sum().item()))
            if k <= 0:
                continue
            _, top_idx = torch.topk(confidence[b], k=k)
            transfer_index[b, top_idx] = True

        x = torch.where(transfer_index, x0, x)

        if output_history:
            history.append(x.clone())

    return _InfillResult(sequences=x, history=history if output_history else [])

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
    ban_repeated_punctuation: bool = False,  # Ban tokens that are 2+ repeated punctuation chars
    ban_crosslingual: bool = False,  # Ban tokens containing non-ASCII characters
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
            "Ċ", "^",
            "Ã", "#", "*",
            "Ĺ", "â","Ģ","¢","Ė","Ī","Ļ","Ĳ","Ď","Ė","Ē",
              "Ŀ","Ń","Ņ","Ŋ","Ŕ","Ŗ","Ş","Ť","Ŧ","Ũ","Ū","Ŭ","Ů","Ű","Ų", "Ŵ","Ŷ","Ÿ","Ź","Ż","Ž", "Ġ", "[","]",
              "<", ">", "{","}","%","^","*","_","+","=","\\","|","~","`", ".\n", "?\n",",\n","!\n", ":\n", ",\n" ,";\n", ")\n",
              "/", "@", "$", "&"
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

        # Rule 3: explicit banned strings — check decoded text only
        # (checking raw vocab strings would ban all Ġ-prefixed tokens since "Ġ" is in extra_banned_strings)
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

        # Rule 7: ban_repeated_punctuation — ban tokens whose non-space content
        # is 2+ punctuation characters (e.g. ',,', '..', '.,', '.,' etc.)
        if ban_repeated_punctuation:
            puncts = [c for c in normalized if not c.isalnum() and not c.isspace()]
            if len(puncts) >= 2:
                banned_ids.add(token_id)
                continue

        # Rule 8: ban_crosslingual — ban tokens with any non-ASCII character
        if ban_crosslingual:
            if any(ord(c) > 127 for c in normalized):
                banned_ids.add(token_id)
                continue

    # -----------------------
    # Rule 7: special tokens
    # -----------------------
    if ban_special_tokens:
        banned_ids.update(tokenizer.all_special_ids)
        # Also ban every token in the added vocabulary (chat-template / tool / fim
        # / vision / mask tokens such as <|im_start|>, <|im_end|>, <|endoftext|>,
        # <|fim_prefix|>, etc.). These are "special" tokens that must not appear
        # in continuous prose; in particular several of them are routinely
        # followed by a newline in the model's training data and would
        # effectively act like pressing Enter mid-paragraph if generated.
        try:
            added_vocab = tokenizer.get_added_vocab()
            banned_ids.update(int(tid) for tid in added_vocab.values())
        except Exception:
            pass

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
            # Pad allowed_mask if model vocab > tokenizer vocab
            mask = allowed_mask.to(logits.device)
            if mask.size(0) < logits.size(2):
                pad = torch.zeros(logits.size(2) - mask.size(0), dtype=torch.bool, device=logits.device)
                mask = torch.cat([mask, pad])
            # Ban all tokens that are not in the allowed list for the first position
            logits[:, 0, ~mask] = float("-inf")
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


def make_position_ban_logits_hook(position_banned_tokens: dict):
    """
    Returns a logits hook that bans specific token IDs at specific sequence positions only.
    Args:
        position_banned_tokens: dict mapping sequence position (int) -> set of token IDs to ban at that position.
    """
    # Pre-compute tensors for efficient masking
    positions = []
    token_ids = []
    for pos, tids in position_banned_tokens.items():
        for tid in tids:
            positions.append(pos)
            token_ids.append(tid)

    def logits_hook(step, x_t, logits):
        # logits: [batch, seq_len, vocab]
        for pos, tid in zip(positions, token_ids):
            logits[:, pos, tid] = float("-inf")
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
    illegal_tokens: Optional[torch.LongTensor] = None,  # Additional tokens to ban globally
    position_banned_tokens: Optional[dict] = None,  # Per-position bans: {seq_pos: {token_id, ...}}
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
        strict_words = getattr(pipeline, "_strict_real_word", False)
        ban_rep_punc = getattr(pipeline, "_ban_repeated_punctuation", False)
        ban_xlang = getattr(pipeline, "_ban_crosslingual", False)
        pipeline._banned_ids = compute_banned_token_ids(
            tok,
            allow_only_alpha=allow_alpha,
            allow_numbers=allow_nums,
            require_real_word=require_words,
            strict_real_word=strict_words,
            ban_repeated_punctuation=ban_rep_punc,
            ban_crosslingual=ban_xlang,
        )

    banned_ids = pipeline._banned_ids
    
    # Merge with additional global illegal tokens if provided
    if illegal_tokens is not None and illegal_tokens.numel() > 0:
        banned_ids = torch.cat([banned_ids.to(illegal_tokens.device), illegal_tokens]).unique()
        print(f"  Banning {illegal_tokens.numel()} additional token(s) globally.")
    
    logits_hook = make_ban_tokens_logits_hook(banned_ids)

    # Add per-position token bans (e.g. ban original token only at its masked position)
    if position_banned_tokens is not None and len(position_banned_tokens) > 0:
        pos_hook = make_position_ban_logits_hook(position_banned_tokens)
        _base_hook = logits_hook
        def _wrap_pos(step, x_t, logits):
            logits = _base_hook(step, x_t, logits)
            logits = pos_hook(step, x_t, logits)
            return logits
        logits_hook = _wrap_pos
        n_pos_bans = sum(len(v) for v in position_banned_tokens.values())
        print(f"  Banning {n_pos_bans} token(s) at {len(position_banned_tokens)} specific position(s).")

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

    # Number of diffusion steps = number of masked tokens (no wasted forward passes)
    n_masked = int((masked_token_tensor == tok.mask_token_id).sum().item())
    n_steps = max(n_masked, 1)  # at least 1 step

    # --- run Dream diffusion ---
    # diffusion_generate_infilling is already bound to model in CustomUnmasker.__init__
    output = model.diffusion_generate_infilling(
        token_tensor=masked_token_tensor,
        attention_mask=attention_tensor,
        max_length=seq_len,
        output_history=True,
        return_dict_in_generate=True,
        steps=n_steps,
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

import math
import random
from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase


# ============================================================================
# SENTIMENT STEERING
# ============================================================================

def _init_sentiment_model(sentiment_model_name, sample_sentiment: Optional[bool] = None):
    """
    Initialize sentiment/emotion model for steering.
    Returns (tokenizer, model) or (None, None) if unavailable.
    """
    if sample_sentiment is None:
        sample_sentiment = False  # Default to False if not explicitly set

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, use_safetensors=True)
        model.eval()

        print(f"Loaded sentiment model: {sentiment_model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"Error: Failed to load sentiment model: {e}")
        if sample_sentiment:
            print("Sentiment validation is active but the model failed to load. Terminating job.")
            import sys
            sys.exit(1)
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


def _compute_sentiment_score(
    text: str,
    sentiment_tokenizer,
    sentiment_model,
    device: torch.device = torch.device("cpu"),
    target_sentiment: str = "positive",
):
    """
    Compute the sentiment score for the given text based on the target sentiment.

    Args:
        text: The input text to evaluate.
        sentiment_tokenizer: Tokenizer for the sentiment model.
        sentiment_model: Sentiment model.
        device: Device to use for computation.
        target_sentiment: Target sentiment, either "positive" or "negative".

    Returns:
        float: The computed sentiment score.
    """
    inputs = sentiment_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

    confidence_positive = probs[1].item()  # Index 1 corresponds to "positive"
    confidence_negative = probs[0].item()  # Index 0 corresponds to "negative"

    if target_sentiment == "positive" or target_sentiment == "POSITIVE":
        return confidence_positive - confidence_negative
    elif target_sentiment == "negative" or target_sentiment == "NEGATIVE":
        return confidence_negative - confidence_positive
    else:
        raise ValueError("Invalid target sentiment. Must be 'positive' or 'negative'.")


def _validate_sentiment(
    final_tokens: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    sentiment_tokenizer,
    sentiment_model,
    device: torch.device = torch.device("cpu"),
    sentiment_target: str = "POSITIVE",
    previous_score: Optional[float] = None,
    alpha_acceptance: float = 100.0,
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
        previous_score: If provided, only accept if current score is BETTER (lower distance).
                            If None, accept first attempt and return its score.
    
    Returns:
        tuple[bool, float]: (validation_passed, score) where score is the distance to target emotion.
                           validation_passed=True if this is first attempt OR score improved over previous_best_score.
    """
     # Decode tokens to text
    text = tokenizer.decode(final_tokens.view(-1).tolist(), skip_special_tokens=True)
    print(f"  Evaluating sentiment for text: \n {text}")
    # Get sentiment probabilities
    sentiment_vector = _compute_sentiment_vector(text, sentiment_tokenizer, sentiment_model, device=device)

    # ---- Look up target label in model's id2label, case-insensitively ----
    if not hasattr(sentiment_model.config, "id2label"):
        raise ValueError("Sentiment model config does not have id2label mapping.")

    labels = [sentiment_model.config.id2label[i].upper() for i in range(len(sentiment_vector))]
    target_upper = sentiment_target.upper()

    if target_upper not in labels:
        raise ValueError(
            f"Sentiment target '{sentiment_target}' not found in model labels. "
            f"Available labels: {labels}"
        )

    score = sentiment_vector[labels.index(target_upper)]

    if previous_score is None:
        return False, score

    # Difference in "goodness"
    acceptance_probability = math.log(score) - math.log(previous_score)
    acceptance_probability *= alpha_acceptance

    # Accept if improved
    if acceptance_probability >= 0:
        print(f"  Sentiment improved: {score:.4f} > {previous_score:.4f}. ✓")
        return True, score
    else:
        # Probabilistic acceptance if worse
        log_random = math.log(random.uniform(0, 1))
        if log_random+1000000 < acceptance_probability: # Disable Metropolis acceptance for debugging
            print(
                f"Sentiment worsened but accepted probabilistically: "
                f"log_random={log_random:.4f} < acceptance_probability={acceptance_probability:.4f}"
                f"New score: {score:.4f}, Previous score: {previous_score:.4f}."
            )
            return True, score
        else:
            print(f"  Sentiment did not improve: Rejected score ({score:.4f}) < Previous score {previous_score:.4f}.")
            return False, previous_score


# ============================================================================
# PERPLEXITY STEERING
# ============================================================================

def _init_perplexity_model(perplexity_model_name: str = "gpt2", validate_perplexity: Optional[bool] = None):
    """
    Initialize a causal LM for perplexity computation.

    Args:
        perplexity_model_name: HuggingFace model name (default: "gpt2").
        validate_perplexity: If True and model fails to load, terminate.

    Returns:
        tuple[tokenizer, model] or (None, None) on failure.
    """
    if validate_perplexity is None:
        validate_perplexity = False

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
        model = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
        model.eval()
        print(f"Loaded perplexity model: {perplexity_model_name}")
        return tokenizer, model
    except Exception as e:
        print(f"Error: Failed to load perplexity model: {e}")
        if validate_perplexity:
            print("Perplexity validation is active but the model failed to load. Terminating job.")
            import sys
            sys.exit(1)
        return None, None


def _compute_perplexity(
    text: str,
    perplexity_tokenizer,
    perplexity_model,
    device: torch.device = torch.device("cpu"),
    max_length: int = 1024,
) -> float:
    """
    Compute the perplexity of *text* under a causal language model.

    Tokens that exceed *max_length* are silently truncated.

    Returns:
        float: Perplexity (≥ 1).  Lower means more fluent / natural text.
    """
    inputs = perplexity_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] < 2:
        # Too short to compute a meaningful CE; return a neutral value
        return 1.0

    with torch.no_grad():
        outputs = perplexity_model(**inputs, labels=input_ids)
        # outputs.loss is mean cross-entropy over tokens
        loss = outputs.loss.item()

    return math.exp(loss)


def _validate_perplexity(
    final_tokens: torch.LongTensor,
    tokenizer: PreTrainedTokenizerBase,
    perplexity_tokenizer,
    perplexity_model,
    device: torch.device = torch.device("cpu"),
    previous_ppl: Optional[float] = None,
    alpha_perplexity: float = 1.0,
) -> tuple[bool, float]:
    """
    Strict acceptance test based on causal-LM perplexity.

    Lower perplexity = more fluent text. A proposal is accepted only when
    its perplexity is no worse than the previously accepted state.

    Args:
        final_tokens: [1, L] or [B, L] token-ID tensor (batch must be 1 for now).
        tokenizer: Diffusion-model tokenizer (used only to decode the text).
        perplexity_tokenizer: Tokenizer for the causal LM.
        perplexity_model: Causal LM.
        device: Torch device.
        previous_ppl: Perplexity of the previous accepted state.
                      If None, accept unconditionally and return the current ppl.
        alpha_perplexity: Unused compatibility argument retained so existing
                          callers do not need to change.

    Returns:
        tuple[bool, float]: (accepted, current_perplexity)
    """
    text = tokenizer.decode(final_tokens.view(-1).tolist(), skip_special_tokens=True)
    print(f"  Evaluating perplexity for text: \n {text}")

    current_ppl = _compute_perplexity(text, perplexity_tokenizer, perplexity_model, device=device)
    print(f"  Perplexity: {current_ppl:.2f}" + (f" (previous: {previous_ppl:.2f})" if previous_ppl is not None else ""))

    if previous_ppl is None:
        # First evaluation – accept unconditionally, record baseline
        return False, current_ppl

    if current_ppl <= previous_ppl:
        print(f"  Perplexity improved: {current_ppl:.2f} ≤ {previous_ppl:.2f}. ✓")
        return True, current_ppl

    print(
        f"  Perplexity worsened and rejected: "
        f"ppl {current_ppl:.2f} > {previous_ppl:.2f}."
    )
    return False, previous_ppl


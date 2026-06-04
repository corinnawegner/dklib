"""
Banned token logic for diffusion unmasking.

Token groups are controlled via boolean flags.  Core groups default to True
(the behaviour used in all production runs so far).  Stricter opt-in groups
default to False.

Parameter reference
-------------------
Core (on by default):
  ban_numbers             -- tokens containing any digit (0-9)
  ban_symbols             -- non-prose punctuation / symbols (#, *, @, /, ...)
  ban_unicode_artifacts   -- garbled byte-level unicode tokens (Ĺ, â, Ģ, ...)
  ban_special_tokens      -- tokenizer.all_special_ids and added-vocab tokens

Stricter (off by default):
  ban_non_alpha             -- only allow pure-letter tokens
  ban_repeated_punctuation  -- tokens with 2+ consecutive punctuation chars (e.g. .., ,,)
  ban_crosslingual          -- tokens with any non-ASCII character after normalisation
  require_real_word         -- only known English dictionary words (needs wordfreq/nltk)
  strict_real_word          -- also ban subword tokens (##...) and single letters

Fine-grained overrides:
  allowed_symbols       -- symbols to keep even when ban_symbols=True
                           (default: . , ! ? ' " : -)
  extra_banned_strings  -- additional substrings that trigger a ban

Slurm variable template
-----------------------
  BAN_NUMBERS=True
  BAN_SYMBOLS=True
  BAN_UNICODE_ARTIFACTS=True
  BAN_NON_ALPHA=False
  BAN_REPEATED_PUNCTUATION=False
  BAN_CROSSLINGUAL=False
  REQUIRE_REAL_WORD=False
  STRICT_REAL_WORD=False
"""

import torch
from typing import Optional, Set
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Token group definitions
# ---------------------------------------------------------------------------

# Non-prose symbols that should never appear in generated English text.
_PROSE_BANNED_SYMBOLS: Set[str] = {
    # HTML
    "<br>", "<br/>",
    # Markdown / code / special display
    "#", "*", "^",
    "•",  # bullet
    "Âł",  # corrupted char
    # Brackets / delimiters
    "[", "]", "<", ">", "{", "}", "(", ")",
    # Operators and special chars
    "%", "_", "+", "=", "\\", "|", "~", "`",
    "/", "@", "$", "&",
    # End-of-text token string
    "<|endoftext|>",
}

# Symbols kept when ban_symbols=True (standard English prose punctuation).
_DEFAULT_ALLOWED_SYMBOLS: Set[str] = {".", ",", "!", "?", "'", '"', ":", "-"}

# Garbled byte-level unicode artifacts produced by some tokenisers.
_UNICODE_ARTIFACTS: Set[str] = {
    "Ã", "â", "Ģ", "¢", "Ė", "Ī", "Ļ", "Ĳ", "Ď", "Ē",
    "Ŀ", "Ń", "Ņ", "Ŋ", "Ŕ", "Ŗ", "Ş",
    "Ť", "Ŧ", "Ũ", "Ū", "Ŭ", "Ů", "Ű", "Ų",
    "Ŵ", "Ŷ", "Ÿ", "Ź", "Ż", "Ž", "Ł",
    "Ġ",  # Ġ  GPT-2 leading-space prefix
}

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_banned_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    # -- Core groups (on by default) -------------------------------------------
    ban_numbers: bool = True,
    ban_symbols: bool = True,
    ban_unicode_artifacts: bool = True,
    ban_special_tokens: bool = True,
    # -- Stricter opt-in groups (off by default) --------------------------------
    ban_non_alpha: bool = False,
    ban_repeated_punctuation: bool = False,
    ban_crosslingual: bool = False,
    require_real_word: bool = False,
    strict_real_word: bool = False,
    # -- Fine-grained overrides ------------------------------------------------
    allowed_symbols: Optional[Set[str]] = None,
    extra_banned_strings: Optional[Set[str]] = None,
) -> torch.LongTensor:
    """Scan the tokenizer vocabulary and return token IDs to ban during generation.
    See module docstring for full parameter reference."""

    if allowed_symbols is None:
        allowed_symbols = _DEFAULT_ALLOWED_SYMBOLS

    # Build combined set of substrings that trigger an automatic ban.
    strings_to_ban: Set[str] = set()
    if ban_symbols:
        strings_to_ban |= _PROSE_BANNED_SYMBOLS - allowed_symbols
    if ban_unicode_artifacts:
        strings_to_ban |= _UNICODE_ARTIFACTS
    if extra_banned_strings:
        strings_to_ban |= extra_banned_strings

    # Initialise optional dictionary for real-word checking.
    word_list = None
    _word_frequency = None
    if require_real_word or strict_real_word:
        try:
            from wordfreq import word_frequency as _wf
            word_list = "wordfreq"
            _word_frequency = _wf
        except ImportError:
            try:
                import nltk
                nltk.download("words", quiet=True)
                from nltk.corpus import words as _nltk_words
                word_list = set(w.lower() for w in _nltk_words.words())
            except (ImportError, LookupError):
                print(
                    "Warning: require_real_word=True but neither 'wordfreq' nor "
                    "'nltk' words corpus is available. Skipping real-word check."
                )
                require_real_word = False
                strict_real_word = False

    # ---------------------------------------------------------------------------
    # Vocabulary scan
    # ---------------------------------------------------------------------------
    banned_ids: Set[int] = set()
    vocab_size = len(tokenizer)

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # Always ban whitespace/newlines -- they break prose output regardless of flags.
        if "\n" in token_str or "\r" in token_str:
            banned_ids.add(token_id)
            continue

        # ban_numbers
        if ban_numbers and any(c.isdigit() for c in token_str):
            banned_ids.add(token_id)
            continue

        # ban_symbols / ban_unicode_artifacts / extra_banned_strings
        if strings_to_ban and any(s in token_str for s in strings_to_ban):
            banned_ids.add(token_id)
            continue

        # Normalise: strip GPT-2 / SentencePiece leading-space prefixes.
        normalized = (
            token_str
            .replace("\u0120", "")  # Ġ  GPT-2 space prefix
            .replace("\u2581", "")  # ▁  SentencePiece prefix
            .replace("\u010a", "")  # Ċ  newline token
            .strip()
        )

        # ban_non_alpha
        if ban_non_alpha and (not normalized or not normalized.isalpha()):
            banned_ids.add(token_id)
            continue

        # require_real_word / strict_real_word -- dictionary check
        if (require_real_word or strict_real_word) and normalized:
            if len(normalized) == 1:
                banned_ids.add(token_id)
                continue
            if word_list == "wordfreq":
                if _word_frequency(normalized.lower(), "en") == 0:
                    banned_ids.add(token_id)
                    continue
            elif isinstance(word_list, set):
                if normalized.lower() not in word_list:
                    banned_ids.add(token_id)
                    continue

        # strict_real_word -- additionally ban subwords and compound tokens.
        if strict_real_word and (token_str.startswith("##") or " " in token_str):
            banned_ids.add(token_id)
            continue

        # ban_repeated_punctuation
        if ban_repeated_punctuation:
            puncts = [c for c in normalized if not c.isalnum() and not c.isspace()]
            if len(puncts) >= 2:
                banned_ids.add(token_id)
                continue

        # ban_crosslingual
        if ban_crosslingual and any(ord(c) > 127 for c in normalized):
            banned_ids.add(token_id)
            continue

    # ---------------------------------------------------------------------------
    # Special tokens
    # ---------------------------------------------------------------------------
    if ban_special_tokens:
        banned_ids.update(tokenizer.all_special_ids)
        # Also ban added-vocabulary tokens (chat-template / mask tokens such as
        # <|im_start|>, <|im_end|>, <|fim_prefix|>, <|mask|>, etc.).
        try:
            added_vocab = tokenizer.get_added_vocab()
            banned_ids.update(int(tid) for tid in added_vocab.values())
        except Exception:
            pass

    return torch.tensor(sorted(banned_ids), dtype=torch.long)
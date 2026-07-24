"""Helpers for driving google/diffusiongemma-*'s own diffusion decoder for masked
infilling, mirroring the role `dklib/dream_helper.py` plays for Dream-org/Dream.

DiffusionGemma is architecturally very different from Dream. Dream is a plain
encoder-only masked LM: you feed it one sequence with some positions replaced by
<mask>, and it fills those positions in directly. DiffusionGemma instead has a causal
autoregressive *encoder* that processes a prompt, and a separate bidirectional
*decoder* that denoises a fixed-length `canvas_length` "canvas" of tokens -- expanding a
fully-random canvas into text over `generation_config.max_denoising_steps` steps
(entropy-threshold acceptance of low-entropy positions + uniform-random renoising of the
rest; see `EntropyBoundSampler` in
`transformers.models.diffusion_gemma.generation_diffusion_gemma`). There is no
`diffusion_generate`/`diffusion_generate_infilling` method on the model, and no absorbing
mask state: unknown canvas positions are natively noised with *uniform random tokens*,
not the tokenizer's `<mask>` id (that id exists on the tokenizer, inherited from the base
Gemma4 vocabulary, but is not what this model's own sampler treats as "unknown").

To reuse the model's own denoising loop for masked infilling of an *existing* text
(instead of driving it through open-ended chat generation), this module:
  - seeds the canvas with the known tokens at their real positions and fresh uniform
    random tokens at masked positions (matching the model's own noise prior), via
    `generate()`'s already-supported `decoder_input_ids=` kwarg (see
    `_prepare_denoiser_inputs` in `generation_diffusion_gemma.py`, and the `generate()`
    docstring: "you can set the starting canvas with `decoder_input_ids`"), and
  - swaps in `PinnedEntropyBoundSampler`, a subclass of the model's own
    `EntropyBoundSampler` that pins the known positions so acceptance/renoising never
    touches them, and reveals exactly one free position per step (the single
    lowest-entropy one) rather than the base class's default of accepting however many
    positions fit under an entropy budget in one step -- matching Dream's one-token-per-
    step reveal (`max_denoising_steps` is auto-raised to cover every free position; see
    `diffusion_generate_infilling`).

`DiffusionGemmaUnmasker` and the ban-token/batch-realization functions below mirror
`dklib/dream_helper.py`'s `CustomUnmasker` + `unmask_batch_dream`, and the way
`experiments/unmasking/run_unmask.py` drives a single u-turn for Dream: load once into an
object exposing `.model`/`.tokenizer`/`.device`/`.model_name`, set ban-token flags as
plain attributes on it (read via `getattr(pipeline, "_ban_numbers", True)`, same defaults
Dream uses), then mask+unmask a batch of Monte Carlo realizations of one sentence in one
call. Unlike Dream, this does NOT go through `dklib.mlm`'s `mask_unmask_monte_batch` /
`_unmask_dispatch` -- DiffusionGemma's canvas-based generation doesn't fit that
per-token-step dispatch loop, and wiring a new branch into that shared file (used by
other Dream/BERT experiments) was deliberately left out of scope here. `mask_unmask_monte_batch`
below is this module's own equivalent, calling `dklib.mlm.prepare_masked_batch` (the same
masking primitive Dream/RoBERTa use) directly.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.generation import LogitsProcessor, LogitsProcessorList
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler,
)

from dklib.banned_tokens import compute_banned_token_ids


class PinnedEntropyBoundSampler(EntropyBoundSampler):
    """`EntropyBoundSampler` variant that (a) pins a fixed set of canvas positions to
    known token ids throughout the denoising loop, and (b) unmasks exactly one *free*
    (non-pinned) position per step -- the single lowest-entropy (most confident) position
    not yet revealed -- rather than the base class's behavior of accepting however many
    positions fit under `entropy_bound` in one step (anywhere from zero to many at once).
    This is the DiffusionGemma analogue of Dream's per-step single-token reveal (see
    `dklib.dream_helper.unmask_batch`/`unmask_batch_dream`, and the module docstring
    below for why `max_denoising_steps` needs to be large enough to cover every free
    position when using this).

    Once revealed, a position stays fixed for the rest of the run (tracked in
    `_revealed_mask`, separate from `known_mask`) -- `entropy_bound` from
    `EntropyBoundSamplerConfig` is unused by this override (kept only because the base
    `__init__` requires a `config` argument).

    `known_tokens`/`known_mask` (both `(batch, canvas_length)`) must be set on the
    instance before generation starts -- `_prepare_sampler` only receives the
    `generation_config`, not per-call canvas info, so `diffusion_generate_infilling`
    below sets them via the model-level `_infill_known_tokens`/`_infill_known_mask`
    attributes consumed by the monkeypatched `_prepare_sampler` (see
    `enable_masked_infilling`).
    """

    known_tokens: torch.LongTensor
    known_mask: torch.BoolTensor
    _revealed_mask: Optional[torch.BoolTensor] = None

    def initialize_canvas(self, batch_size, device):
        canvas = super().initialize_canvas(batch_size, device)
        return torch.where(self.known_mask, self.known_tokens, canvas)

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
        if self._revealed_mask is None:
            self._revealed_mask = torch.zeros_like(self.known_mask)

        # Entropy per canvas position, from THIS step's freshly-predicted logits.
        token_entropy = torch.distributions.Categorical(logits=logits).entropy()

        # Only free (not known/pinned) positions not already revealed are eligible; push
        # everything else to +inf entropy so argmin never picks it.
        eligible = ~self.known_mask & ~self._revealed_mask
        candidate_entropy = torch.where(eligible, token_entropy, torch.full_like(token_entropy, float("inf")))

        # Reveal exactly the single lowest-entropy eligible position per batch row (a
        # row with nothing left eligible reveals nothing -- `has_eligible` guards that).
        newly_revealed = torch.zeros_like(eligible)
        newly_revealed.scatter_(1, candidate_entropy.argmin(dim=-1, keepdim=True), True)
        has_eligible = eligible.any(dim=-1, keepdim=True)
        newly_revealed = newly_revealed & eligible & has_eligible

        self._revealed_mask = self._revealed_mask | newly_revealed
        self.accepted_token_mask = self.known_mask | self._revealed_mask

        accepted_canvas = torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)
        accepted_canvas = torch.where(self.known_mask, self.known_tokens, accepted_canvas)
        return accepted_canvas

    def renoise_canvas(self, accepted_canvas, cur_step):
        renoised_canvas = super().renoise_canvas(accepted_canvas, cur_step)
        return torch.where(self.known_mask, self.known_tokens, renoised_canvas)


def _prepare_sampler_pinned(self, generation_config):
    """Bound onto the model in place of `_prepare_sampler` by `enable_masked_infilling`.

    Only builds a `PinnedEntropyBoundSampler` while `self._infill_mode` is True --
    `diffusion_generate_infilling` toggles that flag on/off (via try/finally) around its
    own `generate()` call. Any *other* `generate()` call on this model (e.g. plain chat
    generation) falls back to the model's normal, unpinned `EntropyBoundSampler` -- this
    monkeypatch is meant to be bound once, permanently, and stay transparent outside of
    actual infilling calls. Without this fallback, a bare `model.generate(...)` call
    would hit `self._infill_known_tokens` before it's ever been set (AttributeError), or
    -- worse -- silently reuse stale known-token state left over from a *previous*
    infilling call.
    """
    if not getattr(self, "_infill_mode", False):
        return EntropyBoundSampler(
            config=generation_config.sampler_config,
            canvas_length=self.config.canvas_length,
            vocab_size=self.config.text_config.vocab_size,
            max_denoising_steps=generation_config.max_denoising_steps,
        )
    sampler = PinnedEntropyBoundSampler(
        config=generation_config.sampler_config,
        canvas_length=self.config.canvas_length,
        vocab_size=self.config.text_config.vocab_size,
        max_denoising_steps=generation_config.max_denoising_steps,
    )
    sampler.known_tokens = self._infill_known_tokens
    sampler.known_mask = self._infill_known_mask
    return sampler


def enable_masked_infilling(model):
    """Binds the pinned-sampler override onto `model`, so any `generate()` call made
    through `diffusion_generate_infilling` pins known canvas positions. Called
    automatically by `DiffusionGemmaUnmasker.__init__`; safe to call again (idempotent
    rebind) if you're managing the model/processor yourself instead."""
    model._prepare_sampler = types.MethodType(_prepare_sampler_pinned, model)
    return model


class DiffusionGemmaUnmasker:
    """Loads DiffusionGemma and exposes the same `.model`/`.tokenizer`/`.device`/
    `.model_name` surface `dklib.dream_helper.CustomUnmasker` exposes for Dream, so u-turn
    driving code can be written the same way for both. Ban-token flags
    (`_ban_numbers`, `_ban_symbols`, etc.) are plain settable attributes, read via
    `getattr(unmasker, "_ban_numbers", True)` inside `diffusion_generate_infilling` --
    same pattern and same effective defaults as `unmask_batch_dream` uses for Dream (see
    `experiments/unmasking/run_unmask.py`, which sets these explicitly after
    construction).
    """

    def __init__(self, model_name: str, device=None, dtype=torch.bfloat16):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if device is None else device,
        )
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device
        enable_masked_infilling(self.model)


class BannedTokensLogitsProcessor(LogitsProcessor):
    """Masks a fixed set of token ids out of the logits at every denoising step -- the
    DiffusionGemma analogue of `dklib.dream_helper.make_ban_tokens_logits_hook` for
    Dream (same role, different call convention: a `LogitsProcessor` for `generate()`'s
    `logits_processor=` list, instead of Dream's `generation_logits_hook_func`)."""

    def __init__(self, banned_token_ids: torch.LongTensor):
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids, scores):
        if self.banned_token_ids.numel() > 0:
            scores[..., self.banned_token_ids.to(scores.device)] = float("-inf")
        return scores


def _get_banned_token_ids(pipeline, tokenizer):
    """Computes (once) and caches on `pipeline` the banned-token set, reading flags off
    `pipeline` via `getattr(..., default)` -- exactly `unmask_batch_dream`'s pattern for
    Dream, including its same effective defaults (note `ban_crosslingual=False`,
    overriding `compute_banned_token_ids`'s own default of `True`, to match what Dream
    actually uses in this project)."""
    if not hasattr(pipeline, "_infill_banned_token_ids"):
        pipeline._infill_banned_token_ids = compute_banned_token_ids(
            tokenizer,
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
    return pipeline._infill_banned_token_ids


def compute_first_token_banned_ids(tokenizer, require_real_word: bool = False) -> torch.LongTensor:
    """Return token ids that should NOT be used as the very first generated token.

    Always bans tokens whose decoded, normalized form doesn't start with an uppercase
    ASCII letter -- so generated text starts like a proper sentence. Direct port of
    `dklib.dream_helper.compute_first_token_banned_ids` for Dream (same normalization,
    same criterion), just scanning DiffusionGemma's own vocabulary.

    If `require_real_word` is True, additionally bans any token whose normalized form
    isn't a real English word per `wordfreq` -- scoped to *just this first-token check*,
    unlike `dklib.banned_tokens.compute_banned_token_ids(require_real_word=True)`, which
    bans every single-character token (including bare "," and ".") everywhere in the
    sequence. Single-character tokens are NOT blanket-banned here, since "I" and "A" are
    themselves valid English words that legitimately start a sentence.
    """
    word_frequency = None
    if require_real_word:
        from wordfreq import word_frequency

    banned = set()
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = token_str.replace("▁", "").replace("Ġ", "").replace("Ċ", "").strip()
        if normalized == "":
            banned.add(token_id)
            continue
        first_char = normalized[0]
        if not (first_char.isalpha() and first_char.isupper() and "A" <= first_char <= "Z"):
            banned.add(token_id)
            continue
        if require_real_word and word_frequency(normalized.lower(), "en") == 0:
            banned.add(token_id)
    return torch.tensor(sorted(banned), dtype=torch.long)


def find_exact_token_id(tokenizer, target_str: str) -> Optional[int]:
    """Scan the vocabulary for the token whose normalized decoded form is exactly
    `target_str` (e.g. "."). Same normalization as `compute_first_token_banned_ids`/
    `dklib.banned_tokens.compute_banned_token_ids`, which is more reliable than
    `tokenizer.encode(target_str)` for a bare punctuation character (encoding can prepend
    a leading-space marker or merge it with adjacent context in ways that don't reflect
    what a *standalone* "." token actually looks like in the vocabulary). Returns None if
    no exact match exists."""
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        normalized = token_str.replace("▁", "").replace("Ġ", "").replace("Ċ", "").strip()
        if normalized == target_str:
            return token_id
    return None


class FirstTokenCapitalizedLogitsProcessor(LogitsProcessor):
    """Bans non-capitalized-start tokens, but only at canvas position 0 -- the
    DiffusionGemma analogue of `dklib.dream_helper.make_first_token_capitalized_logits_hook`
    for Dream. Harmless (if wasted computation) when position 0 is actually a pinned/known
    token during infilling, since `PinnedEntropyBoundSampler` overrides it back to the
    known value regardless of what these logits sample -- it only has real effect when
    position 0 is genuinely being generated (e.g. at/near 100% masking fraction)."""

    def __init__(self, banned_token_ids: torch.LongTensor):
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids, scores):
        if scores.shape[1] > 0 and self.banned_token_ids.numel() > 0:
            scores[:, 0, self.banned_token_ids.to(scores.device)] = float("-inf")
        return scores


def _get_first_token_banned_ids(pipeline, tokenizer, require_real_word: bool = False):
    """Computes (once) and caches on `pipeline` the first-token ban set, separately for
    the capitalization-only vs. capitalization+real-word variants."""
    cache_attr = "_infill_first_token_banned_ids_real_word" if require_real_word else "_infill_first_token_banned_ids"
    if not hasattr(pipeline, cache_attr):
        setattr(
            pipeline, cache_attr,
            compute_first_token_banned_ids(tokenizer, require_real_word=require_real_word),
        )
    return getattr(pipeline, cache_attr)


def _get_period_token_id(pipeline, tokenizer):
    """Computes (once) and caches on `pipeline` the token id for a standalone ".",
    used by `diffusion_generate_infilling(force_last_period=True)`."""
    if not hasattr(pipeline, "_infill_period_token_id"):
        period_id = find_exact_token_id(tokenizer, ".")
        if period_id is None:
            raise ValueError("Tokenizer has no standalone '.' token; can't force_last_period.")
        pipeline._infill_period_token_id = period_id
    return pipeline._infill_period_token_id


class _CanvasHistoryStreamer:
    """Minimal `generate(streamer=...)` object that just records the argmax canvas after
    every denoising step -- the DiffusionGemma equivalent of Dream's
    `output_history=True`/`output.history`. See the `streamer.put_draft(...)` call in
    `DiffusionGemmaGenerationMixin.generate`.
    """

    def __init__(self):
        self.history: list[torch.LongTensor] = []

    def put(self, value):
        pass  # called once with the prompt tokens; not needed here

    def put_draft(self, value):
        self.history.append(value.detach().clone())

    def end(self):
        pass


def diffusion_generate_infilling(
    pipeline: DiffusionGemmaUnmasker,
    masked_token_tensor: torch.LongTensor,
    attention_tensor: torch.Tensor,
    *,
    max_denoising_steps: Optional[int] = None,
    force_first_real_word: bool = False,
    force_last_period: bool = False,
):
    """
    Fills in the masked positions of `masked_token_tensor` using DiffusionGemma's own
    diffusion decoder -- analogous to `dklib.dream_helper.unmask_batch_dream` for Dream,
    and intended to be driven from the same `dklib.mlm.prepare_masked_batch` output used
    for RoBERTa/Dream u-turn experiments.

    Args:
        pipeline: a `DiffusionGemmaUnmasker` (or any object exposing `.model`,
            `.tokenizer`, and optionally the `_ban_*` flags read by
            `_get_banned_token_ids`).
        masked_token_tensor: `(batch, seq_len)`, from `prepare_masked_batch` -- masked
            positions equal `tokenizer.mask_token_id`.
        attention_tensor: `(batch, seq_len)` attention mask (currently unused beyond
            shape/device -- the canvas attends over its full length regardless; kept in
            the signature to match the Dream helper's call shape).
        max_denoising_steps: minimum number of denoising steps to run. Since
            `PinnedEntropyBoundSampler` now reveals exactly one free position per step
            (see its docstring), this is automatically raised to at least the number of
            free positions in the widest batch row -- otherwise, with the model's default
            of 48, any run with more than 48 masked positions would leave the rest stuck
            unresolved at random noise. Pass a larger value to add steps beyond that
            floor (e.g. to give the model idle steps after everything's revealed); it's
            never silently lowered.
        force_first_real_word: if True, position 0 is constrained (via logits banning,
            same mechanism as `_require_capitalized_start`) to real English words, not
            just capitalized tokens. Opt-in and off by default: this only matters when
            position 0 is genuinely free to generate (e.g. at/near 100% masking), and
            shouldn't silently change behavior for ordinary partial-masking u-turn tests.
        force_last_period: if True, the last real content position (`seq_len - 1`) is
            *pinned* (via the same known-token mechanism as real content) to a literal
            "." token, overriding whatever was there originally -- regardless of whether
            it was masked. Opt-in for the same reason as `force_first_real_word`: it
            would otherwise corrupt the fidelity measurement in ordinary u-turn tests by
            injecting a token that was never actually masked/reconstructed.

    Returns:
        filled_token_tensor: `(batch, seq_len)`, `masked_token_tensor` with masked
            positions replaced by the model's reconstruction.
        history: list of `(batch, canvas_length)` tensors, the argmax canvas after each
            denoising step -- feed into `build_diffusiongemma_substitutions` to fill in a
            `dklib.mlm`-style substitutions tensor.
    """
    del attention_tensor  # unused: see docstring

    model = pipeline.model
    tokenizer = pipeline.tokenizer
    device = masked_token_tensor.device
    batch_size, seq_len = masked_token_tensor.shape
    canvas_length = model.config.canvas_length
    if seq_len > canvas_length:
        raise ValueError(
            f"masked_token_tensor has {seq_len} tokens, longer than this model's "
            f"canvas_length={canvas_length}; longer texts need block-wise handling, "
            "not implemented here."
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; can't tell masked positions apart.")

    pad_token_id = model.generation_config.pad_token_id
    vocab_size = model.config.text_config.vocab_size

    # Canvas-shaped known-token state: real content is "known" wherever it isn't the
    # mask token; padding out to canvas_length is "known" too, pinned to pad_token_id so
    # it's never sampled into garbage.
    known_tokens = torch.full((batch_size, canvas_length), pad_token_id, dtype=torch.long, device=device)
    known_tokens[:, :seq_len] = masked_token_tensor
    known_mask = torch.ones((batch_size, canvas_length), dtype=torch.bool, device=device)
    known_mask[:, :seq_len] = masked_token_tensor != tokenizer.mask_token_id

    if force_last_period:
        period_token_id = _get_period_token_id(pipeline, tokenizer)
        known_mask[:, seq_len - 1] = True
        known_tokens[:, seq_len - 1] = period_token_id

    # Seed unknown positions with the model's own noise prior (uniform random tokens,
    # not <mask> -- see module docstring) before handing off as the starting canvas.
    starting_canvas = torch.randint(low=0, high=vocab_size, size=(batch_size, canvas_length), device=device)
    starting_canvas = torch.where(known_mask, known_tokens, starting_canvas)

    model._infill_known_tokens = known_tokens
    model._infill_known_mask = known_mask

    # Minimal encoder "prompt": we want the decoder canvas to be the *actual* content of
    # interest, not a completion appended after a chat prompt, so the encoder side is
    # given nothing more than a single BOS token to prime past_key_values.
    minimal_prompt = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)

    history_streamer = _CanvasHistoryStreamer()
    banned_token_ids = _get_banned_token_ids(pipeline, tokenizer)

    logits_processors = [BannedTokensLogitsProcessor(banned_token_ids)]
    if getattr(pipeline, "_require_capitalized_start", True) or force_first_real_word:
        first_token_banned_ids = _get_first_token_banned_ids(
            pipeline, tokenizer, require_real_word=force_first_real_word,
        )
        logits_processors.append(FirstTokenCapitalizedLogitsProcessor(first_token_banned_ids))

    generate_kwargs = dict(
        input_ids=minimal_prompt,
        decoder_input_ids=starting_canvas,
        max_new_tokens=canvas_length,
        streamer=history_streamer,
        logits_processor=LogitsProcessorList(logits_processors),
        disable_compile=True,
        # The adaptive "stable and confident" stopping criterion averages entropy over
        # the WHOLE canvas. With a mostly-pinned canvas (few masked positions among
        # mostly-known content + padding), that average looks confident almost
        # immediately regardless of whether the actually-masked positions have
        # converged -- disable it so the full step budget always runs.
        stability_threshold=None,
        confidence_threshold=None,
    )
    # PinnedEntropyBoundSampler reveals exactly one free position per step, so there must
    # be at least as many steps as the most free positions any single batch row has --
    # mirrors Dream's own `steps=masked_token_tensor.shape[1]` convention in
    # `dklib.dream_helper.unmask_batch_dream` (generously one step per token slot).
    required_steps = int((~known_mask).sum(dim=1).max().item())
    generate_kwargs["max_denoising_steps"] = max(required_steps, max_denoising_steps or 0, 1)

    model._infill_mode = True
    try:
        with torch.no_grad():
            output = model.generate(**generate_kwargs)
    finally:
        model._infill_mode = False

    final_canvas = output[0][:, -canvas_length:]
    filled_token_tensor = final_canvas[:, :seq_len].clone()

    return filled_token_tensor, history_streamer.history


def build_diffusiongemma_substitutions(substitutions, filled_token_tensor, history):
    """
    Analogous to `dklib.dream_helper.build_dream_substitutions`: fills in the final
    token id and an approximate "unmasking step" for each masked position recorded in
    `substitutions` (from `dklib.mlm.prepare_masked_batch`), using the per-step canvas
    `history` from `diffusion_generate_infilling`.

    Unlike Dream, a canvas position's value here isn't guaranteed to change only once and
    then freeze -- `accept_canvas`/`renoise_canvas` re-evaluate every step, so a position
    can in principle flip more than once before settling. "Unmasking step" is therefore
    defined as the *last* step at which the position still differed from its final value
    (i.e. when it last changed, +1), rather than Dream's "first step at which it equals
    the final value" (which assumes a single, permanent flip).

    Args:
        substitutions: `(batch, max_masks, 4)`, from `prepare_masked_batch`.
        filled_token_tensor: `(batch, seq_len)`, from `diffusion_generate_infilling`.
        history: list of `(batch, canvas_length)` tensors, from
            `diffusion_generate_infilling`.

    Returns:
        substitutions, filled in place.
    """
    for sent_id in range(substitutions.shape[0]):
        for token_unmask in range(substitutions.shape[1]):
            tok_pos = substitutions[sent_id, token_unmask, 0]
            if tok_pos < 0:
                continue
            final_id = filled_token_tensor[sent_id, tok_pos]
            substitutions[sent_id, token_unmask, 2] = final_id
            step_at_unmasking = 0
            for step_idx, canvas in enumerate(history):
                if canvas[sent_id, tok_pos] != final_id:
                    step_at_unmasking = step_idx + 1
            substitutions[sent_id, token_unmask, 3] = step_at_unmasking
    return substitutions


def mask_unmask_monte_batch(
    pipeline, texts, mask_fraction, rng, *,
    max_denoising_steps=None, force_first_real_word=False, force_last_period=False,
):
    """
    Batched single u-turn: mask `mask_fraction` of tokens in each of `texts` (typically
    the same sentence repeated `num_sims` times, for Monte Carlo realizations of one
    mask -> unmask round trip -- same shape of call as
    `dklib.mlm.mask_unmask_monte_batch(sent_batch, unmasker, frac, rng)`, which is what
    `experiments/unmasking/run_unmask.py` calls for Dream's non-sequential/single-u-turn
    case), then fills the masked positions back in with DiffusionGemma's own diffusion
    decoder via `diffusion_generate_infilling`.

    Args:
        pipeline: a `DiffusionGemmaUnmasker`.
        texts: list of strings to mask+unmask (usually `[sentence] * num_sims`).
        mask_fraction: fraction of tokens to mask in each text (see
            `dklib.mlm.prepare_masked_batch`).
        rng: `torch.Generator` for mask-position sampling.
        max_denoising_steps: forwarded to `diffusion_generate_infilling`.
        force_first_real_word, force_last_period: forwarded to
            `diffusion_generate_infilling` -- see its docstring. Both default to False
            (no effect on ordinary partial-masking u-turn fidelity tests).

    Returns:
        masked_token_tensor: `(batch, seq_len)`, the masked input.
        filled_token_tensor: `(batch, seq_len)`, the reconstruction.
        substitutions: `(batch, max_masks, 4)`, filled in via
            `build_diffusiongemma_substitutions`.
    """
    import dklib.mlm as mlm

    masked_token_tensor, attention_tensor, substitutions = mlm.prepare_masked_batch(
        texts, mask_fraction, rng, pipeline.tokenizer, device=pipeline.device,
    )
    filled_token_tensor, history = diffusion_generate_infilling(
        pipeline, masked_token_tensor, attention_tensor,
        max_denoising_steps=max_denoising_steps,
        force_first_real_word=force_first_real_word,
        force_last_period=force_last_period,
    )
    substitutions = build_diffusiongemma_substitutions(substitutions, filled_token_tensor, history)
    return masked_token_tensor, filled_token_tensor, substitutions

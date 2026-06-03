import transformers
import torch
from typing import Optional, Union, Literal
from collections import defaultdict

from .dream_helper import (
    unmask_batch_dream,
    #fill_unmask_steps_from_history,
    #diffusion_generate_infilling,
    CustomUnmasker,
)


def _get_string_to_ids_map(pipeline):
    """
    Build (and cache on the pipeline) a dict mapping each decoded token string
    to the set of all token IDs that decode to that string.
    This handles BPE aliasing: multiple token IDs can produce the same text.
    """
    if hasattr(pipeline, "_string_to_ids_map"):
        return pipeline._string_to_ids_map

    tok = pipeline.tokenizer
    vocab_size = len(tok)
    s2ids = defaultdict(set)
    print("[build_string_to_ids_map] Scanning vocabulary …")
    for tid in range(vocab_size):
        decoded = tok.decode([tid], skip_special_tokens=False)
        s2ids[decoded].add(tid)
    pipeline._string_to_ids_map = dict(s2ids)
    n_aliases = sum(1 for v in s2ids.values() if len(v) > 1)
    print(f"[build_string_to_ids_map] Done – {vocab_size} IDs, {len(s2ids)} unique strings, {n_aliases} with aliases.")
    return pipeline._string_to_ids_map

def _unmask_dispatch(
    masked_token_tensor: torch.LongTensor,
    attention_tensor: torch.Tensor,
    substitutions: torch.LongTensor,
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    rng: Optional[torch.Generator],
    *,
    substitution_step: Optional[int] = None,
    T: float = 1.0,
    dont_predict_special_tokens: bool = True,
    max_kept: int = 100,
    top_token_ids: Optional[torch.LongTensor] = None,
    top_token_probs: Optional[torch.Tensor] = None,
    mask_frac: Optional[float] = None,
    illegal_tokens: Optional[torch.LongTensor] = None,
    position_banned_tokens: Optional[dict] = None,
):
    """
    Unified unmasking entrypoint.
    ALWAYS returns `substitutions`.
    """

    if pipeline.model_name.startswith("inclusionAI/LLaDA") or pipeline.model_name.startswith("Dream-org/Dream"):
        # Diffusion infilling helper returns (tokens, substitutions).
        new_tokens, substitutions = unmask_batch_dream(
            masked_token_tensor,
            attention_tensor,
            substitutions,
            pipeline,
            illegal_tokens=illegal_tokens,
            position_banned_tokens=position_banned_tokens,
        )
        return new_tokens, substitutions

    else:
        assert substitution_step is not None
        unmask_batch(
            masked_token_tensor,
            attention_tensor,
            substitutions,
            pipeline,
            rng,
            substitution_step=substitution_step,
            dont_predict_special_tokens=dont_predict_special_tokens,
            T=T,
            max_kept=max_kept,
            top_token_ids=top_token_ids,
            top_token_probs=top_token_probs,
        )
        return substitutions

def prepare_masked_batch(
    texts: list[str],
    num_masks: Union[int, float],
    rng: torch.Generator,
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    device: torch.device,
    disallowed_ids: Optional[list[int]] = None,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """_summary_
    Takes a list of strings, tokenizes them, and for each one, masks a random subset of the tokens.

    Args:
        texts (list[str]): A list of texts on which to perform masking.
        num_masks (Union[int,float]): The number of tokens to substitute with masks. If a float p between 0,1 is given, then masking is done with probability p.
        rng (torch.Generator): A random number generator for selecting the tokens to mask.
        tokenizer (transformers.tokenization_utils_fast.PreTrainedTokenizerFast): The text tokenizer.
        device (torch.device): which device to store the tokenized tensors on.
        disallowed_ids (Optional[list[int]], optional): A list of additional tokens to ignore -- special tokens are always ignored. Defaults to None.

    Returns:
        tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]: Returns the token id tensor and attention mask, with shapes [batch size x longest sentence], [batch size x longest sentence], and [batch size x longest sentence x 4].
        Substitutions tensor has the following format:
        (i,j,0) = in sentence i, mask number j, which token in the sentence was masked
        (i,j,1) = what was the original token?
        (i,j,2) = -1, but will be used to track final token choice.
        (i,j,3) = -1 but will be used later to track when this mask token was unmasked.

    """
    # Tokens we are not allowed to convert to <mask>!
    if disallowed_ids is None:
        disallowed_ids = []
    disallowed_ids += (
        tokenizer.all_special_ids
    )  # [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    disallowed_ids = torch.tensor(
        disallowed_ids, dtype=torch.int64, device=device
    ).unique()

    # for each row, generate a set of legal tokens to mask.
    # for each row, use rng.choice to choose the appropriate indices to mask.
    # mask those tokens, noting the substitutions made.
    # substitutions should be a [batch_size x num_masks x 3] tensor
    # (i,j,0) = in sentence i, mask # j, which token in the sentence was masked
    # (i,j,1) = what was the original token?
    # (i,j,2) = -1, but will be used to track final token choice.

    mask_id = tokenizer.mask_token_id

    tokenized = tokenizer(texts, padding=True)
    token_tensor = torch.tensor(
        tokenized["input_ids"], dtype=torch.int64, device=device
    )
    attention_tensor = torch.tensor(
        tokenized["attention_mask"], dtype=torch.float32, device=device
    )

    # I will iterate over all of the sentences, because I do not know that the number of allowed tokens in each sentece will be the same
    # This complicates the use of rng.choice(num_allowed_tokens, num_masked, False)
    batch_size = len(texts)

    # now, we need to decide the number of masks for each text.
    num_masks_sent = torch.zeros(batch_size, dtype=torch.int64, device=device)
    if num_masks > 1:
        num_masks = int(num_masks)
    if type(num_masks) == int:
        num_masks_sent[:] = num_masks
    else:
        mask_probability = torch.tensor(num_masks, device=device).float()
        num_masks_sent[:] = torch.sum(
            ~torch.isin(token_tensor[:, :], disallowed_ids), axis=1
        )  # count the number of tokens that are allowed to be masked.
        # print('test: ', torch.binomial(num_masks_sent.float(),mask_probability,generator=rng))
        num_masks_sent[:] = torch.binomial(
            num_masks_sent.float(), mask_probability, generator=rng
        ).long()

    substitutions = torch.zeros(
        (batch_size, torch.max(num_masks_sent), 4), dtype=torch.int64, device=device
    )
    substitutions[:, :, 0] = (
        -1
    )  # to deal with the fact that there may be different numbers of tokens to mask in each sentence, we will substitution rounds with nothing to be -1.
    for sentence_ind in range(batch_size):
        num_masks = num_masks_sent[sentence_ind]
        if num_masks == 0:
            continue
        allowed_tokens_mask = ~torch.isin(token_tensor[sentence_ind, :], disallowed_ids)
        indices = torch.nonzero(allowed_tokens_mask, as_tuple=False).squeeze(-1)

        # forbid first and last valid token
        #
        if indices.numel() > 4: 
            indices = indices[1:-1]
        else:
            indices = indices[:-1] #Three word sentences: Ban only last token 

            # not enough tokens to safely mask anything
         #   continue

        num_masks = min(num_masks, indices.numel())
        if num_masks == 0:
            continue

        # print(indices.shape[0], rng,device)
        subs_inds = torch.arange(num_masks, device=device)
        token_indices_to_mask, _ = torch.sort(
            indices[
                torch.randperm(indices.shape[0], generator=rng, device=device)[
                    subs_inds
                ]
            ]
        )
        # print("num masks: ", num_masks)
        # print("allowed tokens mask: ", allowed_tokens_mask)
        # print("token indices to mask: ", token_indices_to_mask)
        substitutions[sentence_ind, subs_inds, 0] = (
            token_indices_to_mask  # what are the token indices in the original sentence that we are masking?
        )
        substitutions[sentence_ind, subs_inds, 1] = torch.gather(
            token_tensor[sentence_ind, :], 0, token_indices_to_mask
        )  # what are the original token ids ?
        token_tensor[sentence_ind, token_indices_to_mask] = mask_id

    substitutions[:, :, 2] = -1
    substitutions[:, :, 3] = -1
    return token_tensor, attention_tensor, substitutions

def unmask_batch(
    masked_token_tensor: torch.LongTensor,
    attention_tensor: torch.Tensor,
    substitutions: torch.LongTensor,
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    rng: torch.Generator,
    substitution_step: int,
    dont_predict_special_tokens : bool = True, 
    T : float = 1.0,
    max_kept: int = 100,  # added
    top_token_ids: Optional[torch.LongTensor] = None,  # added
    top_token_probs: Optional[torch.Tensor] = None,    # added
):
    """
    Unmasks  single random token from each sentence in the batch, updating the masked_token_tensor in place and updating the substitution tensor.

    Args:
        masked_token_tensor (torch.LongTensor): Batched token tensor.
        attention_tensor (torch.Tensor): Attention mask for the pipeline.
        substitutions (torch.LongTensor): substitutions performed so far.
        pipeline (transformers.pipelines.fill_mask.FillMaskPipeline): unmasking pipeline.
        rng (torch.Generator): Random number generator for choosing the mask token on which to operate.
        substitution_step (int): What step in the substitution chain are we unmasking -- this is noted in substitutions(:,unmasked_index, 3).
        dont_predict_special_tokens (bool): If True, special tokens will not be predicted during unmasking.
        T (float): Temperature for sampling from the unmasking distribution.
        max_kept (int): Maximum number of top tokens to store (optional), utilized only if top_token_ids and top_token_probs are not None.
        top_token_ids (torch.LongTensor): Tensor to store top token IDs (optional).
        top_token_probs (torch.Tensor): Tensor to store top token probabilities (optional).
    """
    logits = pipeline.model.forward(masked_token_tensor, attention_tensor)["logits"]
    batch_size = masked_token_tensor.shape[0]
    illegal_tokens = torch.tensor([], dtype=torch.int64, device=masked_token_tensor.device)
    if(dont_predict_special_tokens):
        illegal_tokens = torch.tensor(pipeline.tokenizer.convert_tokens_to_ids(pipeline.tokenizer.special_tokens_map.values()),dtype = torch.int64, device=masked_token_tensor.device).unique()
    for sent_ind in range(batch_size):
        # print('starting sentence: ', sent_ind)
        masked_token_sub_inds = torch.nonzero(
            (substitutions[sent_ind, :, 2] == -1) & (substitutions[sent_ind, :, 0] >= 0) #extract masked token positions by checking where we have final id = -1 and position not -1
        )
        # print('mask of permitted substitutions: ', (substitutions[sent_ind, :, 2] == -1) & (substitutions[sent_ind,:,0] >= 0))
        # print('masked token sub inds: ',masked_token_sub_inds)
        if masked_token_sub_inds.shape[0] == 0:
            # then, there are no masked tokens remaining in the sentence, and we should continue with another sentence.
            # print(sent_ind, "skipping sentence!")
            continue
        unmask_index = masked_token_sub_inds[ # pick a random position for unmasking
            torch.randint(
                0,
                masked_token_sub_inds.shape[0],
                (1,),
                generator=rng,
                device=rng.device,
            )
        ]
        # print('unmasking token: ',unmask_index, )
        token_index_in_sent = substitutions[sent_ind, unmask_index, 0]
        logits_pre_pmf = logits[sent_ind, token_index_in_sent, :].squeeze()
        if(dont_predict_special_tokens):
            # print('logit shape: ',logits.shape, logits_pre_pmf.shape, illegal_tokens)
            logits_pre_pmf[illegal_tokens] = -1e10 # very small number, so that these tokens are never selected.

        # --- store top tokens ---
        if top_token_ids is not None and top_token_probs is not None:
            probs = logits_pre_pmf.softmax(0)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            kept_ids = sorted_ids[:max_kept]
            kept_probs = sorted_probs[:max_kept]
            top_token_ids[0, unmask_index, :kept_ids.shape[0]] = kept_ids
            top_token_probs[0, unmask_index, :kept_probs.shape[0]] = kept_probs


        if(T == 0):
            new_token_id = torch.argmax(
                logits_pre_pmf.squeeze()
            )  # picking the most likely token
        else:
            new_token_pmf = (
                (logits_pre_pmf.squeeze()/T).softmax(0)
            )  # probability mass function of new tokens, with a temperature.
            new_token_id = torch.multinomial(
                new_token_pmf, 1, False, generator=rng
            )  # sampling a single token
        masked_token_tensor[sent_ind, token_index_in_sent] = substitutions[ # updating masked token tensor in place
            sent_ind, unmask_index, 2
        ] = new_token_id  # performing the substitution
        substitutions[sent_ind, unmask_index, 3] = substitution_step # updating substitution tensor with the unmasking step, not with the final token yet!!

def apply_substitutions(
    token_tensor: torch.LongTensor, substitutions: torch.LongTensor, state="final", sequential=False
) -> None:
    """Applies the mask-unmask substitutions to a token tensor, for instance to see the final text.
        Takes the information from the substitutions tensor and updates the token_tensor.

    Args:
        token_tensor (torch.LongTensor): The token tensor to be transformed, representing the initial input sentences.
        substitutions (torch.LongTensor): The substitution record tensor, shape [num_uturns or num_sents, num_masks, 4]
        state (str): One of 'final' or 'original' -- whether to restor the token tensor to the original state, or to apply the given substitutions.
    Returns:
        None: The token_tensor is modified in place.
    """
    assert (
        token_tensor.shape[0] == substitutions.shape[0]
    )  # ensure the batch sizes are the same.
    assert state in {"final", "original"}
    substitution_index = 2 if state == "final" else 1 # Is the token_tensor in the state before or after the mask-unmask step?

    if not sequential: 
        batch_indices = torch.arange(
            token_tensor.shape[0], device=token_tensor.device
        ).unsqueeze(1)

        # we only want to apply the substitutions that were actually made, so we will mask out the -1 entries in substitutions[:,:,0]
        mask = substitutions[:, :, 0] >= 0
        batch_indices_expanded = batch_indices.expand(-1, substitutions.shape[1])
        valid_batch_indices = batch_indices_expanded[mask]
        valid_token_indices = substitutions[:, :, 0][mask]
        valid_substitution_values = substitutions[:, :, substitution_index][mask]
        token_tensor[valid_batch_indices, valid_token_indices] = valid_substitution_values
    else:  #Substitutions case
        # in the sequential case, we have to do this one sentence at a time, because each sentence may have a different number of substitutions.
        for sent_ind in range(token_tensor.shape[0]):
            mask = substitutions[sent_ind,:,0] >= 0 # determine which tokens have been masked
            token_indices = substitutions[sent_ind, :, 0][mask] # take the token positions of the masked tokens
            substitution_values = substitutions[sent_ind, :, substitution_index][mask] # If 'final', take the token ids after unmasking, if 'original', take the token ids before masking
            token_tensor[sent_ind, token_indices] = substitution_values # Fill the token tensor at the positions where the tokens have been masked (final) or will be masked (original)
            token_tensor[sent_ind+1:, token_indices] = substitutions[sent_ind, :, 2][mask] #making sure we update all the later sentences to reflect the changes made so far.

    # batch_size = token_tensor.shape[0]
    # for sent_ind in range(batch_size):
    #     token_tensor[sent_ind,substitutions[sent_ind,:,0]] = substitutions[sent_ind,:,substitution_index]

def mask_unmask_monte_batch(
    texts: list[str],
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    num_masks: Union[int, float],
    rng: torch.Generator,
    *,
    T: float = 1.0,
    return_tokens: bool = False,
    return_top_tokens: bool = False,
    max_kept: int = 100,
):
    masked_token_tensor, attention_tensor, substitutions = prepare_masked_batch(
        texts, num_masks, rng, pipeline.tokenizer, pipeline.device
    )

    batch_size, max_masks = substitutions.shape[:2]

    top_token_ids = None
    top_token_probs = None
    if return_top_tokens and not pipeline.model_name.startswith("Dream-org/Dream"):
        top_token_ids = torch.zeros(
            (batch_size, max_masks, max_kept),
            dtype=torch.long,
            device=pipeline.device,
        )
        top_token_probs = torch.zeros(
            (batch_size, max_masks, max_kept),
            dtype=torch.float32,
            device=pipeline.device,
        )

    if not pipeline.model_name.startswith("Dream-org/Dream"):
        for step in range(max_masks):
            _unmask_dispatch(
                masked_token_tensor,
                attention_tensor,
                substitutions,
                pipeline,
                rng,
                substitution_step=step,
                T=T,
                max_kept=max_kept,
                top_token_ids=top_token_ids,
                top_token_probs=top_token_probs,
                mask_frac=num_masks if num_masks < 1 else None,
            )

    elif pipeline.model_name.startswith("Dream-org/Dream") or pipeline.model_name.startswith("inclusionAI/LLaDA"):
        _unmask_dispatch(
            masked_token_tensor,
            attention_tensor,
            substitutions,
            pipeline,
            rng=None,
        )

    outputs = [substitutions]
    if return_tokens:
        outputs.append(masked_token_tensor)
    if return_top_tokens and not pipeline.model_name.startswith("Dream-org/Dream"):
        outputs.append((top_token_ids, top_token_probs))

    return tuple(outputs)

def mask_unmask_monte_sequential(
    text: str,
    sequential_iterations: int,
    pipeline,
    num_masks,
    rng,
    *,
    T: float = 1.0,
):
    """
    Performs sequential mask-unmask on a single text, for a given number of iterations.
    
    If grammar or sentiment validation is enabled and fails, re-runs unmasking with the same
    masked input (up to max_retries).
    """
    from dklib.dream_helper import _validate_grammar, _validate_sentiment, _init_grammar_checker, _validate_perplexity
    
    # --- prepare initial masked sentence---
    masked_token_tensor, attention_tensor, substitutions = prepare_masked_batch(
        [text]*sequential_iterations,
        num_masks,
        rng,
        pipeline.tokenizer,
        pipeline.device,
    )

    max_masks = masked_token_tensor.shape[1]

    # Get validation settings
    validate_grammar = getattr(pipeline, "_validate_grammar", False)
    sample_sentiment = getattr(pipeline, "_sample_sentiment", False)
    validate_perplexity = getattr(pipeline, "_validate_perplexity", False)
    max_unmasking_retries = getattr(pipeline, "_validation_max_retries", 5)
    
    # Initialize grammar checker if needed
    if validate_grammar and not hasattr(pipeline, "_grammar_checker"):
        grammar_method = getattr(pipeline, "_grammar_method", "gpt")
        from dklib.dream_helper import _init_grammar_checker
        pipeline._grammar_checker = _init_grammar_checker(grammar_method)
    
    # Get the sentimen for the original paragraph
    if sample_sentiment:
        
        sentiment_tokenizer = getattr(pipeline, "_sentiment_tokenizer", None)
        sentiment_model = getattr(pipeline, "_sentiment_model", None)
        sentiment_target = getattr(pipeline, "_sentiment_target", "POSITIVE")
        tokenized = pipeline.tokenizer([text], padding=True)
        token_tensor = torch.tensor(
            tokenized["input_ids"], dtype=torch.int64, device=pipeline.device
        )
        validation_passed, new_score = _validate_sentiment(
            token_tensor,
            pipeline.tokenizer,
            sentiment_tokenizer,
            sentiment_model,
            device=pipeline.device,
            sentiment_target=sentiment_target,
            previous_score = 10e-12
        )

    # Compute baseline perplexity for the original paragraph (only when sentiment steering is also active)
    if sample_sentiment and validate_perplexity:
        perplexity_tokenizer = getattr(pipeline, "perplexity_tokenizer", None)
        perplexity_model_obj = getattr(pipeline, "perplexity_model", None)
        alpha_perplexity = getattr(pipeline, "_alpha_perplexity", 1.0)
        tokenized = pipeline.tokenizer([text], padding=True)
        token_tensor = torch.tensor(
            tokenized["input_ids"], dtype=torch.int64, device=pipeline.device
        )
        _, current_ppl = _validate_perplexity(
            token_tensor,
            pipeline.tokenizer,
            perplexity_tokenizer,
            perplexity_model_obj,
            device=pipeline.device,
            previous_ppl=None,
            alpha_perplexity=alpha_perplexity,
        )

    fail_counter = 0

    # Initialize last_validated_tokens with the original (unmasked) sentence
    # This tracks the last state that passed validation, so we can revert to it if needed
    initial_tokens = masked_token_tensor[0].clone()
    initial_mask_positions = substitutions[0, :, 0]
    initial_valid_mask = initial_mask_positions >= 0
    initial_tokens[initial_mask_positions[initial_valid_mask]] = substitutions[0, initial_valid_mask, 1]
    last_validated_tokens = initial_tokens.unsqueeze(0)
    last_uturn_token_ids = last_validated_tokens.clone()  # [1, seq_len] – the token IDs from the previous u-turn

    for uturn in range(sequential_iterations):

        # For each uturn, we need to unmask the previously masked tokens, fill in the substitutions, and then re-mask for the next uturn.
        step_tokens = masked_token_tensor[uturn, :].unsqueeze(0)
        step_att = attention_tensor[uturn, :].unsqueeze(0)
        step_subs = substitutions[uturn, :].unsqueeze(0) # [1, M, 4], filled with -1s at start, except for initial masking step
        
        mask_positions = step_subs[0, :, 0]
        valid_mask = mask_positions >= 0

        position_banned_tokens = None

        # Unmask
        if pipeline.model_name.startswith("Dream-org/Dream") or pipeline.model_name.startswith("inclusionAI/LLaDA"):
            #print(f"Using Dream unmasking... (attempt {unmasking_attempt + 1}/{max_unmasking_retries})")
            unmasked_tokens, step_subs = _unmask_dispatch(
                step_tokens,
                step_att,
                step_subs,
                pipeline,
                rng=None,
                position_banned_tokens=position_banned_tokens,
            )
        else:
            for step in range(max_masks):
                _unmask_dispatch(
                    step_tokens,
                    step_att,
                    step_subs,
                    pipeline,
                    rng,
                    substitution_step=step,
                    T=T,
                )
            # step_tokens is updated in-place by unmask_batch; expose it as
            # unmasked_tokens so the code below (which is shared with the
            # Dream path) can use it unconditionally.
            unmasked_tokens = step_tokens.clone()

        # Given the filled substitution tensor, update masked_token_tensor with the unmasked token ids
        apply_substitutions(step_tokens, step_subs, state="final")

        # Copy step_subs back to substitutions (step_subs is a new tensor due to unsqueeze)
        substitutions[uturn] = step_subs.squeeze(0)

        # Save  
        save_unmasked_tokens = unmasked_tokens

        #print(f" NEW TOKEN IDs: {unmasked_tokens}")

        # --- Validate unmasked tokens ---
        validation_passed = True

        # Check grammar if enabled
        """
        if validate_grammar:
            use_previous_uturn_step = False
            use_last_attempt = True
            grammar_checker = getattr(pipeline, "_grammar_checker", None)
            if grammar_checker is not None:
                print(f"  Checking grammar...")
                print("Candidate: ", unmasked_tokens)
                if not _validate_grammar(unmasked_tokens, pipeline.tokenizer, grammar_checker, device=step_tokens.device):
                    print(f"  Grammar validation failed, retrying unmasking...")
                    validation_passed = False

                    # Get the token that was generated at the masked position
                    mask_pos = step_subs[0, 0, 0].item()  # Position of the single masked token
                    failed_token_id = unmasked_tokens[0, mask_pos].item()
                    illegal_tokens_for_uturn.add(failed_token_id)
                    print(f"  Banning token {failed_token_id} at position {mask_pos}")

                    # Go back to previous unmasked state and re-mask
                    step_tokens = previous_unmasked_tokens.clone()
                    step_tokens[0, mask_pos] = pipeline.tokenizer.mask_token_id
                    step_subs = original_step_subs.clone()

                    unmasking_attempt += 1
                    continue
        """

        # Check sentiment if enabled
        if sample_sentiment:

            previous_score = new_score

            if sentiment_tokenizer is not None and sentiment_model is not None: # Else go to default sentiment model
                print(f"Checking sentiment...")
                validation_passed, sentiment_score = _validate_sentiment(
                    unmasked_tokens,
                    pipeline.tokenizer,
                    sentiment_tokenizer,
                    sentiment_model,
                    device=step_tokens.device,
                    sentiment_target=sentiment_target,
                    previous_score = previous_score
                )

                if validation_passed: # If the sentiment validation passed, update the sentiment score
                    new_score = sentiment_score

        # Check perplexity if enabled – only when sentiment steering is active
        if sample_sentiment and validate_perplexity and validation_passed:
            if perplexity_tokenizer is not None and perplexity_model_obj is not None:
                print(f"Checking perplexity...")
                ppl_passed, new_ppl = _validate_perplexity(
                    unmasked_tokens,
                    pipeline.tokenizer,
                    perplexity_tokenizer,
                    perplexity_model_obj,
                    device=step_tokens.device,
                    previous_ppl=current_ppl,
                    alpha_perplexity=alpha_perplexity,
                )
                if ppl_passed:
                    current_ppl = new_ppl
                else:
                    validation_passed = False

        if not validation_passed:
            fail_counter += 1
            # Fill with the unmasked text from the last validated state
            print(f"  Could not pass validation. Reverting to last validated state.")
            # Go back to the last validated tokens
            unmasked_tokens = last_validated_tokens.clone()
            # Overwrite the substitution tensor: set final token ids (col 2) to the tokens from last validated state
            mask_positions = substitutions[uturn, :, 0]
            valid_mask = mask_positions >= 0
            substitutions[uturn, valid_mask, 2] = last_validated_tokens[0, mask_positions[valid_mask]]
        else:
            # Validation passed, update last_validated_tokens to current state
            last_validated_tokens = unmasked_tokens.clone()

        # Store current token ids for banning in the next u-turn
        last_uturn_token_ids = unmasked_tokens.clone()

        # --- re-mask masked_token_tensor for next u-turn and prepare substitution tensor ---
        if uturn < sequential_iterations-1:
            masked_token_tensor[uturn+1] = unmasked_tokens #.squeeze(0)
            subs_mask = substitutions[uturn+1,:,0] > 0 # Mask new masked token tensor at correct positions
            # Store the current unmasked tokens as 'original' tokens for the next u-turn BEFORE masking
            substitutions[uturn+1,subs_mask,1] = unmasked_tokens[0,substitutions[uturn+1,subs_mask,0]]
            # Now apply the mask
            masked_token_tensor[uturn+1,substitutions[uturn+1,subs_mask,0]] = pipeline.tokenizer.mask_token_id
            # So substitutions[uturn+1] now has the original token ids (before masking) at positions that will be masked

    #print("shape of substitutions after mask_unmask_monte_sequential:", all_substitutions.shape) # torch.Size([10, 189, 4])

    # Remove unnecessary lines: adjust all_substitutions[1] according to the actual maximum number of masks across each uturn step, deleting lines in which all values are -1
    print(f"Final fail count: {fail_counter} out of {sequential_iterations} u-turns.")
    return substitutions

def reconstruct_sequential_tensor_texts(initial_text, substitutions, pipeline):
    token_tensor = torch.tensor(pipeline.tokenizer.encode(initial_text, add_special_tokens=True)).unsqueeze(0).to(substitutions.device)
    token_tensor = token_tensor.repeat(substitutions.shape[0],1)
    apply_substitutions(token_tensor, substitutions, state='final',sequential=True)
    return token_tensor

def mask_all_single(
    text: str, pipeline: transformers.pipelines.fill_mask.FillMaskPipeline
) -> tuple[torch.LongTensor, torch.DoubleTensor]:
    """
    Masks each token in the text, then performs inference on that masked token.

    Args:
        text (str): The source text for masking.
        pipeline (transformers.pipelines.fill_mask.FillMaskPipeline): The unmasking pipeline.

    Returns:
        torch.LongTensor: The token ids that were masked in the sentence.
        torch.DoubleTensor: The output logits from each masked token.
    """
    tokenizer = pipeline.tokenizer
    mask_id = tokenizer.mask_token_id
    tokenized_text = tokenizer(text, return_tensors="pt")
    # sending the token tensors to the appropriate device.
    tokens = tokenized_text["input_ids"].to(pipeline.device)
    # here, we duplicate the tokenized text N-2 times, because we don't want to mask the start and end of sentence tokens.
    tokenized_replicates = tokens.reshape(1, -1).repeat(tokens.size(1) - 2, 1)
    # masking along the diagonal:
    tokenized_replicates[
        torch.arange(0, tokenized_replicates.shape[0]),
        torch.arange(1, tokens.size(1) - 1),
    ] = mask_id
    # building the attention mask after sending the attention mask to the appropriate device.
    att = tokenized_text["attention_mask"].to(pipeline.device)
    attention_replicates = att.expand(tokenized_replicates.shape)
    # computing the logits for the masked tokens:
    # print('\ntokenized replicates shape: ', tokenized_replicates.shape)
    with torch.no_grad():
        logits = pipeline.model.forward(tokenized_replicates, attention_replicates)["logits"]
    masked_logits = logits[
        torch.arange(0, tokenized_replicates.shape[0]),
        torch.arange(1, tokens.size(1) - 1),
        :,
    ]
    return tokens[0, 1:-1], masked_logits

"""Shared helpers for loading Hugging Face LLM backbones."""

from transformers import AutoModel


def load_llm(
    model_name_or_path,
    *,
    device=None,
    eval_mode=True,
    trust_remote_code=True,
    **from_pretrained_kwargs,
):
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        **from_pretrained_kwargs,
    )
    if device is not None and from_pretrained_kwargs.get("device_map") is None:
        model = model.to(device)
    if eval_mode:
        model = model.eval()
    return model
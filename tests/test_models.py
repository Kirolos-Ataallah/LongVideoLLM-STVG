import pytest

from stvg.config import load_config
from stvg.models.qwen_vl_vllm import sampling_kwargs


def test_vllm_config_inherits_inputs_from_hf_config():
    hf = load_config("vstar_qwen3_5_4b").model
    vllm = load_config("vstar_qwen3_5_4b_vllm").model
    assert vllm.type == "qwen_vl_vllm" and vllm.name == "qwen3_5_4b_vllm"
    # Everything that defines what the model sees must be identical across backends.
    for key in ("model_path", "video", "video_reader", "chat_template_kwargs", "generation"):
        assert vllm[key] == hf[key], key


def test_sampling_kwargs_translates_transformers_generation():
    assert sampling_kwargs({"max_new_tokens": 16, "do_sample": False}) == {
        "max_tokens": 16,
        "temperature": 0.0,
    }
    assert sampling_kwargs({"max_new_tokens": 8, "do_sample": True, "top_p": 0.9}) == {
        "max_tokens": 8,
        "top_p": 0.9,
    }
    with pytest.raises(ValueError):
        sampling_kwargs({"num_beams": 4})

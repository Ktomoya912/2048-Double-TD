import torch
import pytest


def test_cnn_deep_policy_output_shape():
    from models.CNN_DEEP_POLICY import Model
    model = Model()
    x = torch.zeros(4, 99)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"


def test_cnn_deep_policy_output_is_logits():
    """Output should be raw logits (no softmax), allowing CrossEntropyLoss."""
    from models.CNN_DEEP_POLICY import Model
    model = Model()
    x = torch.randn(2, 99)
    out = model(x)
    # Raw logits can be any float, not bounded to [0,1]
    assert out.dtype == torch.float32


def test_cnn_deep_multi_output_shape():
    from models.CNN_DEEP_MULTI import Model
    model = Model()
    x = torch.zeros(4, 99)
    value, pi = model(x)
    assert value.shape == (4, 1), f"Expected (4,1), got {value.shape}"
    assert pi.shape == (4, 4), f"Expected (4,4), got {pi.shape}"


def test_alpha_zero_state_output_shape():
    from models.ALPHA_ZERO_STATE import Model
    model = Model()
    x = torch.zeros(2, 99)
    value, pi = model(x)
    assert value.shape == (2, 1), f"Expected (2,1), got {value.shape}"
    assert pi.shape == (2, 4), f"Expected (2,4), got {pi.shape}"

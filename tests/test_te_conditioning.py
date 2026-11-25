"""engine.te_conditioning: the parts shared by the three engines for keeping the text
encoder's conditioning in main memory (台帳 §1-299). CPU only.

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_te_conditioning.py
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from engine.te_conditioning import device_copy, host_copy, tensor_sha256  # noqa: E402


class Pair(NamedTuple):
    a: object
    b: object


COPIES = [host_copy, lambda v: device_copy(v, "cpu")]


@pytest.mark.parametrize("copy", COPIES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_copy_keeps_value_dtype_shape_and_is_new(copy, dtype):
    src = (torch.arange(24).reshape(2, 3, 4) % 7).to(dtype)
    keep = src.clone()
    out = copy(src)
    assert out.dtype == dtype and out.shape == src.shape and torch.equal(out, src)
    assert out.data_ptr() != src.data_ptr()
    src.zero_()
    assert torch.equal(out, keep)          # the source's change does not reach the copy
    out.fill_(5)
    assert not torch.equal(src, out) and int(src.abs().sum()) == 0   # nor the other way


def test_host_copy_detaches():
    src = torch.ones(3, requires_grad=True) * 2
    assert src.requires_grad
    assert not host_copy(src).requires_grad


@pytest.mark.parametrize("copy", COPIES)
def test_containers_and_other_values(copy):
    assert copy(None) is None
    t = torch.ones(2)
    out = copy((t, None, 3, "s"))
    assert type(out) is tuple and out[1] is None and out[2] == 3 and out[3] == "s"
    assert out[0] is not t and torch.equal(out[0], t)
    out = copy([t, t])
    assert type(out) is list and out[0] is not t and out[0] is not out[1]
    out = copy(Pair(t, None))
    assert type(out) is Pair and out.b is None and out.a is not t and torch.equal(out.a, t)
    nested = copy(Pair([t, (t, Pair(t, 1))], {"k": t}))
    assert type(nested) is Pair and type(nested.a) is list and type(nested.a[1]) is tuple
    assert type(nested.a[1][1]) is Pair and nested.a[1][1].b == 1
    assert nested.a[0] is not t and nested.a[1][0] is not t and nested.a[1][1].a is not t
    assert nested.b == {"k": t} or nested.b["k"] is t   # a dict is not a container here: as it is
    dev = torch.device("cpu")
    assert copy(dev) is dev and copy(7) == 7 and copy("x") == "x"


def test_inference_tensor_is_copied_hashed_outside_the_mode():
    with torch.inference_mode():
        src = torch.arange(10, dtype=torch.float32).to(torch.bfloat16)
    assert src.is_inference()
    out = host_copy(src)
    assert torch.equal(out, src)
    assert tensor_sha256(out) == tensor_sha256(src)
    assert len(tensor_sha256(out)) == 64


def test_tensor_sha256_bytes_and_identity_with_the_h3_name():
    for t in (torch.arange(6).reshape(2, 3).to(torch.bfloat16), torch.arange(6, dtype=torch.int64).reshape(2, 3)):
        raw = t.view(torch.int16 if t.element_size() == 2 else torch.int64).numpy().tobytes()
        assert tensor_sha256(t) == hashlib.sha256(raw).hexdigest()
    pytest.importorskip("comfy_kitchen")
    from engine_h3 import pipeline_h3
    assert pipeline_h3.tensor_sha256 is tensor_sha256

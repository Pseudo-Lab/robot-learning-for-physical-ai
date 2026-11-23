from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

from conftest import PROJECT_ROOT, requested_device


NOTEBOOK_ROOT = PROJECT_ROOT / "hw1_pytorch_tutorial/src"


def notebook_namespace(name: str, *, stop_before: int | None = None) -> dict:
    """Execute only definition/example cells, excluding dataset download cells."""
    path = NOTEBOOK_ROOT / f"{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    module_name = f"notebook_{name}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for index, cell in enumerate(notebook["cells"]):
        if stop_before is not None and index >= stop_before:
            break
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{path}#cell-{cell.get('id', index)}", "exec"), namespace)
    return namespace


def test_tensor_shapes_masks_and_attention_on_requested_device() -> None:
    ns = notebook_namespace("ex1")
    device = requested_device()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    torch.testing.assert_close(
        ns["sum_over_dim"](x, 1), torch.tensor([3.0, 7.0], device=device)
    )
    counts = ns["batched_token_histogram"](
        torch.tensor([[0, 1, 1], [2, 2, 0]], device=device), 3
    )
    torch.testing.assert_close(
        counts, torch.tensor([[1, 2, 0], [1, 0, 2]], device=device)
    )
    logits = torch.zeros(1, 1, 3, 3, device=device)
    masked = ns["apply_causal_mask"](logits)
    assert masked[0, 0, 0, 1] < -1e8
    assert masked[0, 0, 2, 1] == 0
    q = torch.randn(2, 3, 4, 5, device=device)
    scores = ns["einsum_qk_scores"](q, q)
    assert scores.shape == (2, 3, 4, 4)


def test_autograd_dataset_optimizer_and_initialization() -> None:
    ns = notebook_namespace("ex2")
    x = torch.tensor([1.5, -2.0])
    expected = 2 * x
    torch.testing.assert_close(ns["grad_with_autograd_grad"](x), expected)
    torch.testing.assert_close(ns["grad_with_backward"](x), expected)
    padded = ns["pad_1d_sequences"](
        [torch.tensor([1, 2, 3]), torch.tensor([4])], pad_value=0
    )
    assert padded.tokens.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert padded.padding_mask.tolist() == [[False, False, False], [False, True, True]]

    parameter = torch.tensor([1.0], requires_grad=True)
    state = ns["init_adamw_state"](parameter)
    ns["adamw_step_"](parameter, torch.tensor([0.5]), state, lr=0.1)
    assert parameter.item() < 1.0
    layer = torch.nn.Linear(4, 3)
    ns["init_linear_"](layer, "zero")
    assert torch.count_nonzero(layer.weight) == 0


def test_neural_network_layers_match_reference_and_backpropagate() -> None:
    ns = notebook_namespace("ex3", stop_before=14)
    device = requested_device()
    custom = ns["Linear"](4, 3).to(device)
    reference = torch.nn.Linear(4, 3).to(device)
    with torch.no_grad():
        reference.weight.copy_(custom.weight)
        reference.bias.copy_(custom.bias)
    x = torch.randn(5, 4, device=device, requires_grad=True)
    torch.testing.assert_close(custom(x), reference(x))
    output = ns["MLP"](4, 8, 2, depth=3, use_layernorm=True).to(device)(x)
    assert output.shape == (5, 2)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    normalized = ns["RMSNorm"](4).to(device)(torch.ones(3, 4, device=device))
    torch.testing.assert_close(
        normalized, torch.ones_like(normalized), atol=1e-5, rtol=1e-5
    )


def test_mnist_accuracy_keeps_starter_call_and_weights_the_last_batch() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    ns = notebook_namespace("ex3", stop_before=14)
    path = NOTEBOOK_ROOT / "ex3.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    ns["DataLoader"] = DataLoader
    for index in (16, 18, 19):
        cell = notebook["cells"][index]
        exec(compile("".join(cell["source"]), str(path), "exec"), ns)

    device = requested_device()
    model = torch.nn.Linear(2, 2, bias=False, device=device)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2, device=device))
    ns["model"] = model
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    assert ns["accuracy"](loader) == 2 / 3
    loss, _ = ns["classifier_metrics"](model, loader)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(torch.tensor(loss), expected)
    losses = ns["train_classifier"](model, loader, loader, 1e-3, 1)
    assert type(losses) is list and len(losses) == 2


def test_vit_ffn_glu_forward_backward_and_parameter_budget() -> None:
    ns = notebook_namespace("ex4", stop_before=9)
    device = requested_device()
    image = torch.randn(4, 1, 28, 28, device=device)
    parameter_counts = {}
    for kind in ("ffn", "geglu", "swiglu"):
        model = ns["TinyViT"](4, 32, 4, 1, 64, 0.0, kind).to(device)
        logits = model(image)
        assert logits.shape == (4, 10)
        loss = logits.square().mean()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
        parameter_counts[kind] = sum(
            parameter.numel() for parameter in model.parameters()
        )
    assert parameter_counts["geglu"] == parameter_counts["swiglu"]
    assert parameter_counts["geglu"] < parameter_counts["ffn"] * 1.15

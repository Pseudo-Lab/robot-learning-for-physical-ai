"""Check each exercise on 100 independently generated fixtures against PyTorch and analytic answers."""
import copy
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(os.environ["R"]) / "tests"))
from test_hw1 import notebook_namespace

close = torch.testing.assert_close


def tensor_case(fn, seed):
    torch.manual_seed(seed)
    n, d, t = torch.randint(2, 8, (3,)).tolist()
    x = torch.randn(n, d)
    v = torch.randn(d)
    mask = torch.rand(n, d) > .4
    mask[0] = True
    mask[-1] = False
    tokens = torch.randint(0, d, (n, t))
    idx = torch.randint(0, n, (n + 1,))
    y = x.clone()
    y[-1, 0] = -5
    seq = torch.randn(n, t, d)
    q, k, val = torch.randn(3, n, 2, t, d)
    scores = q @ k.transpose(-2, -1)
    causal = torch.arange(t)[None, :] > torch.arange(t)[:, None]
    weights = scores.softmax(-1)
    diag = min(n, d)
    check = lambda name, expected, *args: close(fn[name](*args), expected)
    check("make_tensor", x.double(), x.tolist(), torch.float64)
    check("make_zeros", torch.zeros(n, d), (n, d))
    check("make_ones_like", torch.ones(n, d), x)
    check("make_arange", torch.tensor(list(range(-d, n, 2))), -d, n, 2)
    check("make_linspace", torch.linspace(-1, 2, n), -1, 2, n)
    check("make_randn", torch.randn((n, d), generator=torch.Generator().manual_seed(seed)), (n, d), seed)
    check("cast_dtype_and_move", x.double(), x, torch.device("cpu"), torch.float64)
    check("reshape_tensor", x.flatten(), x, (-1,))
    check("view_tensor", x.flatten(), x, (-1,))
    check("flatten_from_dim", seq.reshape(n, -1), seq, 1)
    check("add_singleton_dim", x[:, None, :], x, 1)
    check("remove_singleton_dims", x, x[:, None, :], 1)
    check("transpose_last_two", x.T, x)
    check("permute_bhwc_to_bchw", q.movedim(-1, 1), q)
    contiguous = fn["make_contiguous"](x.T)
    assert contiguous.is_contiguous()
    close(contiguous, x.T)
    check("slice_rows", x[:-1], x, 0, n-1)
    check("select_columns", x[:, [d-1, 0]], x, [d-1, 0])
    check("get_diagonal", x[torch.arange(diag), torch.arange(diag)], x)
    original = x.clone()
    check("set_subtensor", y, x, -1, 0, -5)
    close(x, original)
    check("gather_rows", x[idx], x, idx)
    check("sum_over_dim", x.T.sum(0), x, 1)
    check("mean_over_dim", x.sum(1) / d, x, 1)
    maxv, maxi = fn["max_over_dim"](x, 1)
    close(maxv, x.amax(1))
    close(maxi, x.argmax(1))
    check("argmax_over_dim", maxi, x, 1)
    check("broadcast_add_vector", x + v[None, :], x, v)
    check("concat_tensors", x.repeat(2, 1), [x, x], 0)
    check("stack_tensors", x[:, None, :].expand(-1, 2, -1), [x, x], 1)
    check("repeat_tensor", torch.cat([x, x], 1), x, (1, 2))
    check("expand_tensor", torch.stack([v]*n), v, n, d)
    check("cumsum_over_dim", torch.stack([x[:, :i+1].sum(1) for i in range(d)], 1), x, 1)
    check("where_select", torch.where(mask, x, -x), mask, x, -x)
    check("one_hot", torch.eye(d)[tokens], tokens, d, torch.float32)
    values = torch.randn(len(idx))
    check("scatter_add_1d", torch.stack([values[idx == i].sum() for i in range(n)]), values, idx, n)
    check("batched_token_histogram", torch.stack([torch.bincount(row, minlength=d) for row in tokens]), tokens, d)
    check("masked_mean", torch.stack([a[m].mean() if m.any() else x.new_zeros(()) for a,m in zip(x,mask)]), x, mask, 1)
    matrix = torch.randn(d, n)
    check("einsum_linear_btd_dh_to_bth", seq @ matrix, seq, matrix)
    check("einsum_pairwise_dot", (seq * seq).sum(-1), seq, seq)
    check("einsum_qk_scores", scores, q, k)
    check("einsum_apply_attention", weights @ val, weights, val)
    check("stable_softmax", (x + 1000).softmax(1), x + 1000, 1)
    check("masked_fill_tensor", torch.where(mask, -9, x), x, mask, -9)
    masked = torch.zeros_like(x)
    for i in range(n):
        if (~mask[i]).any():
            masked[i, ~mask[i]] = x[i, ~mask[i]].softmax(0)
    check("masked_softmax", masked, x, mask, 1)
    check("make_causal_mask", causal, t)
    check("apply_causal_mask", torch.where(causal, -1e9, scores), scores)
    return dict(rows=n, features=d, sequence=t)


def training_case(fn, seed):
    torch.manual_seed(seed)
    n, d, t = torch.randint(2, 9, (3,)).tolist()
    a, b = torch.randn(2, n, d)
    close(fn["grad_with_autograd_grad"](a), 2*a)
    close(fn["grad_with_backward"](a), 2*a)
    ga, gb = fn["grad_wrt_multiple_inputs"](a, b)
    close(ga, 2*a+b)
    close(gb, a)
    tokens = torch.randint(0, 50, (n, t+1))
    pairs = fn["TensorPairDataset"](a, b)
    next_token = fn["NextTokenDataset"](tokens)
    assert len(pairs) == len(next_token) == n
    crops = [fn["RandomCropSequenceDataset"](tokens, t-1, seed) for _ in range(2)]
    batch = []
    for i in range(n):
        x, y = pairs[i]
        close(x, a[i])
        close(y, b[i])
        x, y = next_token[i]
        close(x, tokens[i, :-1])
        close(y, tokens[i, 1:])
        crop = crops[0][i]
        close(crop, crops[1][i])
        assert any(torch.equal(crop, window) for window in tokens[i].unfold(0, t-1, 1))
        length = 1+i%t
        batch.append((x[:length], y[:length]))
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(item) for item in inputs])
    packed = fn["pad_1d_sequences"](list(inputs), -1)
    close(packed.tokens, pad_sequence(inputs, batch_first=True, padding_value=-1))
    close(packed.lengths, lengths)
    close(packed.padding_mask, packed.tokens == -1)
    collated = fn["collate_next_token_batch"](batch, -1)
    close(collated["input_ids"], packed.tokens)
    close(collated["target_ids"], pad_sequence(targets, batch_first=True, padding_value=-1))
    close(collated["attention_mask"], ~packed.padding_mask)
    close(collated["padding_mask"], packed.padding_mask)
    batches = list(fn["make_dataloader"](pairs, 2, shuffle=False))
    close(torch.cat([p[0] for p in batches]), a)
    close(torch.cat([p[1] for p in batches]), b)
    params = [a.clone(), b.clone()]
    refs = [nn.Parameter(p.clone()) for p in params]
    states = [fn["init_adamw_state"](p) for p in params]
    opt = torch.optim.AdamW(refs, lr=.002, betas=(.8,.95), weight_decay=.03)
    for iteration in range(4):
        grads = [torch.randn_like(p) for p in params]
        for p,g in zip(refs, grads):
            p.grad = g.clone()
        opt.step()
        fn["adamw_step_many_"](params, grads, states, .002, (.8,.95), weight_decay=.03)
        for p, ref, state in zip(params, refs, states):
            close(p, ref)
            close(state.m, opt.state[ref]["exp_avg"])
            close(state.v, opt.state[ref]["exp_avg_sq"])
            assert state.t == iteration+1
    model = nn.Linear(d,d)
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=.03)
    opt_ref = torch.optim.SGD(ref.parameters(), lr=.03)
    loss = fn["train_step_mse"](model, (a,b), opt)
    expected = F.mse_loss(ref(a), b)
    close(torch.tensor(loss), expected)
    expected.backward()
    opt_ref.step()
    for p,q in zip(model.parameters(),ref.parameters()):
        close(p,q)
    weight = torch.empty(n,d,2,3)
    assert fn["fan_in_fan_out"](weight) == (d*6,n*6)
    for name, init in (("xavier_uniform_", nn.init.xavier_uniform_), ("kaiming_uniform_", nn.init.kaiming_uniform_)):
        torch.manual_seed(seed)
        fn[name](weight)
        torch.manual_seed(seed)
        close(weight, init(torch.empty_like(weight)))
    for scheme in ("xavier","kaiming_relu","zero"):
        torch.manual_seed(seed)
        fn["init_linear_"](model, scheme)
        torch.manual_seed(seed)
        expected = torch.empty_like(model.weight)
        if scheme == "zero":
            expected.zero_()
        elif scheme == "xavier":
            nn.init.xavier_uniform_(expected)
        else:
            nn.init.kaiming_uniform_(expected)
        close(model.weight, expected)
        close(model.bias, torch.zeros_like(model.bias))
    return dict(rows=n, features=d, sequence=t+1)


for index, case in enumerate((tensor_case, training_case), 1):
    functions = notebook_namespace(f"ex{index}")
    rows = []
    for seed in range(107500,107600):
        try:
            rows.append(dict(seed=seed, success=True, **case(functions,seed)))
        except Exception as error:
            import traceback
            rows.append(dict(seed=seed, success=False, error=traceback.format_exc()))
    with open(f"ex{index}.json","x") as output:
        json.dump(dict(total=len(rows),success=sum(row["success"] for row in rows),rows=rows),output,indent=2)
    print(index, sum(row["success"] for row in rows), next((r["error"] for r in rows if not r["success"]), ""))

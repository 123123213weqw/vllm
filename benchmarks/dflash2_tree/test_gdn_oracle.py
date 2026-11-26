# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from benchmarks.dflash2_tree.gdn_oracle import (
    GDNState,
    commit_path,
    gdn_step,
    verify_tree,
)

# Both leaves are at equal depth; the second path is not a row prefix.
PARENTS = (-1, 0, 0, 1, 2)
PATHS = ((0,), (0, 1), (0, 2), (0, 1, 3), (0, 2, 4))


@pytest.fixture
def example():
    gen = torch.Generator(device="cpu").manual_seed(13)

    def sample(*shape):
        return torch.randn(*shape, generator=gen, device="cpu", dtype=torch.float64)

    prefix = GDNState(sample(7, 3), sample(3, 2))
    # Five tree rows plus one fresh token for the continuation check.
    return prefix, sample(6, 7), sample(7, 4), -sample(6).abs(), sample(6).sigmoid()


def serial_path(prefix, projected, weight, log_decay, beta):
    """Score one path with a whole-sequence convolution and delta operator."""
    sequence = torch.cat((prefix.conv_history, projected.T), dim=-1)
    mixed = F.silu(F.conv1d(sequence[None], weight[:, None], groups=weight.shape[0]))
    q, k, v = mixed.squeeze(0).T.split((2, 2, 3), dim=-1)
    q = q / (q.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt() / 2**0.5
    k = k / (k.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()
    state = prefix.recurrent
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    outputs = []
    for i in range(len(projected)):
        erase = eye - beta[i] * torch.outer(k[i], k[i])
        state = log_decay[i].exp() * (state @ erase) + beta[i] * torch.outer(v[i], k[i])
        outputs.append(state @ q[i])
    return torch.stack(outputs), GDNState(sequence[:, -3:], state)


def assert_state_close(actual, expected):
    torch.testing.assert_close(actual.conv_history, expected.conv_history)
    torch.testing.assert_close(actual.recurrent, expected.recurrent)


def tree_round(example):
    prefix, projected, weight, log_decay, beta = example
    return verify_tree(PARENTS, prefix, projected[:5], weight, log_decay[:5], beta[:5])


def test_tree_matches_independent_serial_paths(example):
    outputs, states = tree_round(example)
    prefix, projected, weight, log_decay, beta = example
    for row, path in enumerate(PATHS):
        ids = list(path)
        expected, state = serial_path(
            prefix, projected[ids], weight, log_decay[ids], beta[ids]
        )
        torch.testing.assert_close(outputs[row], expected[-1])
        assert_state_close(states[row], state)


def test_changing_one_sibling_does_not_leak_to_the_other(example):
    prefix, projected, weight, log_decay, beta = example
    outputs, states = tree_round(example)
    perturbed = projected.clone()
    perturbed[1] += 2
    changed, changed_states = tree_round((prefix, perturbed, weight, log_decay, beta))
    torch.testing.assert_close(changed[[0, 2, 4]], outputs[[0, 2, 4]])
    assert_state_close(changed_states[4], states[4])
    # Positive control: the perturbed branch's descendant changes.
    assert not torch.allclose(changed[3], outputs[3])


@pytest.mark.parametrize("accepted", [(0,), (0, 2), (0, 2, 4)])
def test_continuation_from_committed_path(example, accepted):
    prefix, projected, weight, log_decay, beta = example
    _, states = tree_round(example)
    committed = commit_path(PARENTS, states, accepted)
    got, state = gdn_step(committed, projected[5], weight, log_decay[5], beta[5])
    ids = [*accepted, 5]
    expected, expected_state = serial_path(
        prefix, projected[ids], weight, log_decay[ids], beta[ids]
    )
    torch.testing.assert_close(got, expected[-1])
    assert_state_close(state, expected_state)


def test_commit_rejects_a_path_across_siblings(example):
    _, states = tree_round(example)
    with pytest.raises(ValueError, match="parent links"):
        commit_path(PARENTS, states, (0, 1, 4))

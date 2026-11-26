# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GDNState:
    conv_history: torch.Tensor
    recurrent: torch.Tensor


def gdn_step(
    state: GDNState,
    projected_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, GDNState]:
    """Run one CPU reference step with a single GDN head.

    The convolution input has K + K + V channels in Q/K/V order. The
    history has shape `[K + K + V, width - 1]`, oldest first, and the
    recurrent state has shape `[V, K]`. Gates are already transformed,
    i.e. log_decay is g and beta is sigmoid(b). No input is mutated.
    """
    value_dim, key_dim = state.recurrent.shape
    window = torch.cat((state.conv_history, projected_qkv[:, None]), dim=-1)
    q, k, v = F.silu((window * conv_weight).sum(dim=-1)).split(
        (key_dim, key_dim, value_dim)
    )
    q = q * (q.square().sum() + 1e-6).rsqrt() * key_dim**-0.5
    k = k * (k.square().sum() + 1e-6).rsqrt()
    decayed = state.recurrent * log_decay.exp()
    recurrent = decayed + torch.outer(beta * (v - decayed @ k), k)
    return recurrent @ q, GDNState(window[:, 1:].clone(), recurrent)


def verify_tree(
    parents: Sequence[int],
    prefix: GDNState,
    projected_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, list[GDNState]]:
    """Return each row's output and post-row state without committing.

    Row 0 is the unprocessed anchor (parent -1); its input state contains
    only the committed prefix. Every other row must follow its parent.
    """
    if not parents or parents[0] != -1:
        raise ValueError("row 0 must be the anchor")
    if not (len(parents) == len(projected_qkv) == len(log_decay) == len(beta)):
        raise ValueError("provide inputs for every row, including the anchor")

    outputs, states = [], []
    for row, parent in enumerate(parents):
        if row and not 0 <= parent < row:
            raise ValueError("each non-anchor row must follow its parent")
        output, state = gdn_step(
            states[parent] if row else prefix,
            projected_qkv[row],
            conv_weight,
            log_decay[row],
            beta[row],
        )
        outputs.append(output)
        states.append(state)
    return torch.stack(outputs), states


def commit_path(
    parents: Sequence[int], states: Sequence[GDNState], rows: Sequence[int]
) -> GDNState:
    """Snapshot the state after the last accepted row, including the anchor.

    The next output token chosen by the target (bonus or mismatch) is not
    a verified row and must not appear here. Stop/length handling can pass
    a prefix of the accepted path.
    """
    if not rows or rows[0] != 0:
        raise ValueError("accepted rows must start with the anchor")
    for parent, child in zip(rows[:-1], rows[1:]):
        if not parent < child < len(parents) or parents[child] != parent:
            raise ValueError("accepted rows must follow parent links")
    state = states[rows[-1]]
    return GDNState(state.conv_history.clone(), state.recurrent.clone())

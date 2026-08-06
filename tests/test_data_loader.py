from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from bird_song.data import make_loader


def test_training_shuffle_is_reproducible_with_an_explicit_seed() -> None:
    dataset = TensorDataset(torch.arange(20))

    first = torch.cat([batch[0] for batch in make_loader(dataset, 4, 0, training=True, seed=123)])
    second = torch.cat([batch[0] for batch in make_loader(dataset, 4, 0, training=True, seed=123)])
    different = torch.cat([batch[0] for batch in make_loader(dataset, 4, 0, training=True, seed=456)])

    assert torch.equal(first, second)
    assert not torch.equal(first, different)

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_cache(tmp_path: Path):
    for domain, n in (("d1__t", 6), ("d2__t", 4)):
        for split, count in (("train", n - 2), ("val", 1), ("test", 1)):
            torch.save({
                "features": torch.randn(count, 4, 1024),
                "cls": torch.randn(count, 4, 1024),
                "episode_ids": torch.arange(count),
                "domain": domain, "split": split,
            }, tmp_path / f"{domain}_{split}.pt")


def test_sample_shape_and_view(tmp_path):
    from mlvla.meta.e2e.evidence import EvidenceBank
    _make_cache(tmp_path)
    bank = EvidenceBank(tmp_path, ["d1__t", "d2__t"])
    g = torch.Generator().manual_seed(0)
    x = bank.sample("d1__t", k=3, generator=g)
    assert x.shape == (3, 4, 1024)
    v = bank.view("d2__t")
    assert v.shape == (7,)


def test_split_ids(tmp_path):
    from mlvla.meta.e2e.evidence import EvidenceBank
    _make_cache(tmp_path)
    bank = EvidenceBank(tmp_path, ["d1__t"])
    assert bank.split_episode_ids("d1__t", "val") == [0]


def test_domain_mean_deterministic(tmp_path):
    from mlvla.meta.e2e.evidence import EvidenceBank
    _make_cache(tmp_path)
    bank = EvidenceBank(tmp_path, ["d1__t"])
    a = bank.domain_mean("d1__t", "val")
    assert torch.equal(a, bank.domain_mean("d1__t", "val"))
    assert a.shape[1:] == (4, 1024)

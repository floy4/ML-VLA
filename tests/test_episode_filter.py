# tests/test_episode_filter.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_intersect_none_keeps_all():
    from mlvla.experts.train import intersect_episodes
    assert intersect_episodes([3, 1, 2], None) == [1, 2, 3]


def test_intersect_filters_and_sorts():
    from mlvla.experts.train import intersect_episodes
    assert intersect_episodes([5, 1, 3, 9], {1, 3, 7}) == [1, 3]


def test_intersect_empty_raises():
    import pytest
    from mlvla.experts.train import intersect_episodes
    with pytest.raises(ValueError, match="empty"):
        intersect_episodes([4, 5], {1, 2})

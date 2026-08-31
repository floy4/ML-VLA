from .direct_ab import DirectABHyperNetwork

__all__ = ["DirectABHyperNetwork", "FiLMABHyperNetwork"]


def __getattr__(name: str):
    # Lazy import: film_hypernet imports hypernet.direct_ab, so importing it eagerly here
    # creates a circular import when film_hypernet itself is imported first.
    if name == "FiLMABHyperNetwork":
        from mlvla.meta.film_hypernet import FiLMABHyperNetwork

        return FiLMABHyperNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .direct_ab import DirectABHyperNetwork

__all__ = ["DirectABHyperNetwork", "FiLMABHyperNetwork", "SharedFiLMABHyperNetwork", "V4DirectABHyperNetwork"]


def __getattr__(name):
    # Lazy import: film_hypernet imports hypernet.direct_ab, so importing it eagerly here
    # creates a circular import when film_hypernet itself is imported first.
    lazy = {
        "FiLMABHyperNetwork": ("mlvla.meta.film_hypernet", "FiLMABHyperNetwork"),
        "SharedFiLMABHyperNetwork": ("mlvla.meta.hypernet.shared_head", "SharedFiLMABHyperNetwork"),
        "V4DirectABHyperNetwork": ("mlvla.meta.hypernet.v4_head", "V4DirectABHyperNetwork"),
    }
    if name in lazy:
        import importlib
        module = importlib.import_module(lazy[name][0])
        return getattr(module, lazy[name][1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

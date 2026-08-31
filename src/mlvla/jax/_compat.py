"""Keep openpi's unused PyTorch backend out of a pure-JAX process."""

from __future__ import annotations

import sys
import types


def isolate_unused_pytorch_backend() -> None:
    """Work around openpi.models.model importing PI0Pytorch unconditionally.

    JAX execution never calls the PyTorch restoration branch.  A lightweight
    placeholder avoids importing Transformers (and therefore avoids forcing a
    tokenizers package change) while preserving the rest of openpi unchanged.
    """
    name = "openpi.models_pytorch.pi0_pytorch"
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

    # openpi.models.tokenizer imports AutoProcessor for the FAST/PyTorch path,
    # while π0.5 JAX uses its SentencePiece PaligemmaTokenizer exclusively.
    # Keep that unused import from enforcing a tokenizers package downgrade.
    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _UnusedAutoProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError(
                    "AutoProcessor belongs to the isolated FAST/PyTorch path; "
                    "the WIZARD π0.5 JAX pipeline uses PaligemmaTokenizer"
                )

        class _UnusedPreTrainedTokenizerBase:
            """Marker needed by Hugging Face datasets' serialization check."""

        transformers.AutoProcessor = _UnusedAutoProcessor
        transformers.PreTrainedTokenizerBase = _UnusedPreTrainedTokenizerBase
        sys.modules["transformers"] = transformers

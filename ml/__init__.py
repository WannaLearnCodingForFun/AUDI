"""AUDI ML components for persistent semantic memory."""

__all__ = ["SemanticMemoryEngine"]


def __getattr__(name: str):
    if name == "SemanticMemoryEngine":
        from ml.semantic_memory import SemanticMemoryEngine

        return SemanticMemoryEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

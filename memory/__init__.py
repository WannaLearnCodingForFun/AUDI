"""Memory storage package for AUDI."""

__all__ = ["LocalSemanticMemoryStore", "VectorMemory"]


def __getattr__(name: str):
    if name == "LocalSemanticMemoryStore":
        from memory.local_store import LocalSemanticMemoryStore

        return LocalSemanticMemoryStore
    if name == "VectorMemory":
        from memory.vector_memory import VectorMemory

        return VectorMemory
    raise AttributeError(f"module {name!r} has no attribute {name!r}")

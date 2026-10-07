"""Text utilities shared by training and inference (standard library only)."""

from .normalize import contains_khmer, normalize_text

__all__ = ["contains_khmer", "normalize_text"]

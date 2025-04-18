"""
Dimensional Cascade Distillation Package.

This package provides tools for distilling high-dimensional embeddings to lower 
dimensions while preserving similarity relationships, supporting dimensional
cascade search approaches.
"""

from src.distillation.models import DimensionDistiller, CascadeDistiller
from src.distillation.trainer import DistillationTrainer

__all__ = [
    'DimensionDistiller',
    'CascadeDistiller',
    'DistillationTrainer',
] 
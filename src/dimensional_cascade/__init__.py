"""
Dimensional Cascade: Multi-Resolution Semantic Search
====================================================

A progressive dimensionality reduction approach to semantic search,
enabling ultra-fast initial retrieval with increasingly precise refinement.
"""

from dimensional_cascade.core import DimensionalCascade, CascadeConfig
from dimensional_cascade.models import ModelHierarchy
from dimensional_cascade.search import CascadeSearch

__version__ = "0.1.0" 
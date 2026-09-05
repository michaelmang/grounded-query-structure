"""Query concept extraction for hybrid search — never retrieves or answers."""

from .heuristic import HeuristicStructurer
from .models import QueryStructure, StructureError
from .openai_provider import OpenAIStructurer
from .structurer import QueryStructurer

__all__ = [
    "HeuristicStructurer",
    "OpenAIStructurer",
    "QueryStructure",
    "QueryStructurer",
    "StructureError",
]

__version__ = "0.1.0"

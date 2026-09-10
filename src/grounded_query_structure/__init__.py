"""Topic-map match + bounded query expansion for hybrid search — never retrieves or answers."""

from .cache import FileStructureCache
from .heuristic import HeuristicStructurer
from .models import QueryStructure, StructureError, TopicEntry, TopicMatch
from .openai_provider import OpenAIStructurer
from .structurer import QueryStructurer
from .topic_map import TopicMap

__all__ = [
    "FileStructureCache",
    "HeuristicStructurer",
    "OpenAIStructurer",
    "QueryStructure",
    "QueryStructurer",
    "StructureError",
    "TopicEntry",
    "TopicMap",
    "TopicMatch",
]

__version__ = "0.2.0"

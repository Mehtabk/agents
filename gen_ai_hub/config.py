from enum import Enum
import os


class GenAiHubEndpoint(Enum):
    DEV = os.getenv("DEV_URL")
    PRD = os.getenv("PRD_URL")


class LanguageModel(Enum):
    GPT_4O = "gpt-4o"
    GPT_4 = "gpt-4"
    GPT_4O_MINI = "gpt-4o-mini"
    O3_MINI = "o3-mini"


class ApiVersion(Enum):
    V2024_06_01 = "2024-06-01"


class EmbeddingModel(Enum):
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

from enum import Enum


class ColumnType(Enum):
    INTEGER = "int"
    DATETIME = "datetime"
    FLOAT = "float"
    STRING = "str"


class ContentType(Enum):
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    COMPLEX = "complex"
    RENDER = "render"


class BenchmarkColumns(Enum):
    SESSION_ID = ("session_id", ColumnType.INTEGER)
    BEGIN_TIME = ("BT", ColumnType.DATETIME)
    FIRST_TOKEN_TIME = ("FT", ColumnType.DATETIME)
    LAST_TOKEN_TIME = ("LT", ColumnType.DATETIME)
    PROMPT_TOKENS = ("prompt_tokens", ColumnType.INTEGER)
    PROMPT_MS = ("prompt_ms", ColumnType.FLOAT)
    PROMPT_PER_TOKEN_MS = ("prompt_per_token_ms", ColumnType.FLOAT)
    PROMPT_PER_SECOND = ("prompt_per_second", ColumnType.FLOAT)
    PREDICTED_MS = ("predicted_ms", ColumnType.FLOAT)
    PREDICTED_PER_TOKEN_MS = ("predicted_per_token_ms", ColumnType.FLOAT)
    PREDICTED_PER_SECOND = ("predicted_per_second", ColumnType.FLOAT)
    TIME_TO_FIRST_TOKEN = ("TTFT", ColumnType.FLOAT)
    TIME_GENERATION_TIME = ("TGT", ColumnType.FLOAT)
    PP = ("PP", ColumnType.FLOAT)
    ERROR = ("error", ColumnType.STRING)
    CONTENT_SAMPLE = ("content_sample", ColumnType.STRING)

    @classmethod
    def get_all_columns(cls):
        return [(column.value[0], column.value[1].value) for column in cls]

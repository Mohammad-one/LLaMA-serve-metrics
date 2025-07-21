from enum import Enum

from transformers import AutoTokenizer


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
    COMPLETION_TOKENS = ("completion_tokens", ColumnType.INTEGER)
    PROMPT_MS = ("prompt_ms", ColumnType.FLOAT)
    PROMPT_PER_TOKEN_MS = ("prompt_per_token_ms", ColumnType.FLOAT)
    PROMPT_PER_SECOND = ("prompt_per_second", ColumnType.FLOAT)
    PREDICTED_MS = ("predicted_ms", ColumnType.FLOAT)
    PREDICTED_PER_TOKEN_MS = ("predicted_per_token_ms", ColumnType.FLOAT)
    PREDICTED_PER_SECOND = ("predicted_per_second", ColumnType.FLOAT)
    TIME_TO_FIRST_TOKEN = ("TTFT", ColumnType.FLOAT)
    TIME_GENERATION_TIME = ("TGT", ColumnType.FLOAT)
    PP = ("PP", ColumnType.FLOAT)
    TG = ("TG", ColumnType.FLOAT)
    ERROR = ("error", ColumnType.STRING)

    @classmethod
    def get_all_columns(cls):
        return [(column.value[0], column.value[1].value) for column in cls]


class graphic_card(Enum):
    RTX5000 = "RTX5000"
    RTX4060 = "RTX4060"


class LlmModel(Enum):
    QWEN_15B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    QWEN_06B = "Qwen/Qwen3-0.6B"

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.value)


class QuestionsEnum(Enum):
    COLOR_OF_THE_SKY = "What is the color of the sky?"
    MATH_EXPRESSION = "2 + 3 * 4 / 6 is ?"
    HUMAN_EYES = "How many eyes does a human have?"
    DOGS_FLY = "Can dogs fly?"
    DAYS_IN_A_WEEK = "How many days are in a week?"
    OCEAN_COLOR = "What is the color of the ocean on a map?"
    TWO_PLUS_TWO = "What is 2 plus 2?"
    CAPITAL_OF_FRANCE = "What is the capital of France?"
    CONTINENTS = "How many continents are there?"
    MAIN_INGREDIENT_OF_YOGURT = "What is the main ingredient in yogurt?"

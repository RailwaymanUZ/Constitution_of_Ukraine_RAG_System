import os
import dotenv
from loguru import logger

logger.add("logs/app.log", format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_DOCUMENTS_IN_RAG = 40
RETURN_DOCUMENT_IN_RAG = 20
RETURN_FINAL_DOCUMENT = 18
PARAM_MMR = 0.9
MODEL_LLM = "o4-mini-2025-04-16"
CROSS_ENCODER_MODEL = "intfloat/multilingual-e5-small"

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# Dataset files
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"


# Saved embedding files
EMBEDDING_PATHS = {
    "bert": PROCESSED_DATA_DIR / "X_bert.npy",
    "sbert": PROCESSED_DATA_DIR / "X_sbert.npy",
    "e5": PROCESSED_DATA_DIR / "X_infloat.npy",
    "bge": PROCESSED_DATA_DIR / "X_baai.npy",
    "fasttext": PROCESSED_DATA_DIR / "X_fasttext.npy",
}


# Saved vector models
VECTOR_MODEL_PATHS = {
    "word2vec": PROCESSED_DATA_DIR / "word2vec.kv",
    "glove": PROCESSED_DATA_DIR / "glove.kv",
    "fasttext": PROCESSED_DATA_DIR / "fasttext.kv",
}


# Pretrained model identifiers
EMBEDDING_MODELS = {
    "bert": "bert-base-uncased",
    "sbert": "all-MiniLM-L6-v2",
    "e5": "intfloat/e5-small-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "word2vec": "word2vec-google-news-300",
    "glove": "glove-wiki-gigaword-100",
    "fasttext": "fasttext-wiki-news-subwords-300",
}


# Zero-shot classification model
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_MODEL_DIR = MODELS_DIR / "bart-large-mnli"

HR_LABEL = "This job title is related to human resources."
NON_HR_LABEL = "This job title is not related to human resources."
ZERO_SHOT_LABELS = [HR_LABEL, NON_HR_LABEL]
HR_CONFIDENCE_THRESHOLD = 0.5


# Search configuration
DEFAULT_QUERY = "aspiring human resources"
SECONDARY_QUERY = "seeking human resources"

SIMILARITY_THRESHOLD = 0.3
STAR_CONFIDENCE_THRESHOLD = 0.9
MAX_DUPLICATE_COUNT = 2
TOP_STARRED_CANDIDATES = 10
ALPHA = 0.5


# E5 and BGE formatting
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    "word2vec": 300,
    "glove": 100,
    "fasttext": 300,
}


# Evaluation configuration
EVALUATION_K = 10
MANUAL_EVALUATION_COUNT = 30


# Ensure required directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
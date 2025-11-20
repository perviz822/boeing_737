import os
from dotenv import load_dotenv


def load_env():
    """
    Load required environment variables.

    Inputs:
        None

    Returns:
        dict with HUGGING_FACE_TOKEN, DEEP_SEEK_API_KEY, PERSIST_DIR
    """
    load_dotenv()
    names = ["HUGGING_FACE_TOKEN", "DEEP_SEEK_API_KEY", "PERSIST_DIR"]

    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError(f"FATAL: Missing env variables: {missing}")

    return {name: os.environ[name] for name in names}

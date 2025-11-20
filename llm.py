import os
from pydantic import SecretStr
from langchain_openai import ChatOpenAI


def create_llm(env):
    """
    Create the DeepSeek-chat LLM client.

    Inputs:
        env (dict): must contain "DEEP_SEEK_API_KEY"

    Returns:
        ChatOpenAI instance
    """

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=SecretStr(env["DEEP_SEEK_API_KEY"]),
        temperature=0.0,
    )

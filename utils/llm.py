import os
import dotenv
from typing import List

from openai import AzureOpenAI, AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled

# Load environment variables from the specifaied file
dotenv.load_dotenv(".env")

def set_openai_config():
    set_default_openai_client(get_oai_client())
    set_tracing_disabled(True)

def get_oai_client():
    api_key = os.getenv("AOAI_KEY")
    endpoint = os.getenv("AOAI_ENDPOINT")

    if not api_key:
        raise ValueError("AOAI_KEY environment variable is required")
    if not endpoint:
        raise ValueError("AOAI_ENDPOINT environment variable is required")

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{endpoint}openai/v1/"
        )
        return client
    except Exception as e:
        raise Exception(f"Failed to create OpenAI client: {str(e)}")

def get_aoai_client():
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AOAI_ENDPOINT"),
            api_key=os.getenv("FOUNDRY_KEY"),
            api_version=os.getenv("AOAI_API_VERSION")
        )
        return client
    except KeyError as e:
        print("Missing required model_config key: %s", e)
        raise

def embed_text(text: str) -> List[float]:
    response = get_aoai_client().embeddings.create(
        input=text,
        model=os.getenv("embeddingModel")
    )
    return response.data[0].embedding

def invoke_llm(system: str, user: str, model: str) -> str:
    response = get_aoai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    text = "What is the capital of France?"
    print(embed_text(text))
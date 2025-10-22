import os
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

# Configuration for local GGUF model
# you need to download the model from huggingface and place it in the project root folder for this to work
MODEL_PATH = os.getenv("MODEL_PATH", "../Llama-3.2-3B-Instruct-IQ3_M.gguf")
N_CTX = int(os.getenv("N_CTX", "2048"))  # Context window size
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))  # Number of layers to offload to GPU (0 = CPU only)

# Initialize the model (singleton pattern)
_llm = None

def get_llm():
    """Get or initialize the Llama model instance."""
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False
        )
    return _llm

async def chat_completion(messages, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Generates a chat completion using the local GGUF model via llama-cpp-python.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated response text
    """
    try:
        llm = get_llm()

        # Generate response using llama-cpp-python's chat completion format
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        # Extract the content from the response
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {str(e)}")

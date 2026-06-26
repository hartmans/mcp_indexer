import logging
import os

# Set env var to avoid some init warnings/errors
os.environ["OLLAMA_API_KEY"] = "ollama"

try:
    from gptme.llm.models import get_model
    
    model_name = "ollama/gemma4:31b"
    meta = get_model(model_name)
    
    print(f"Model: {meta.model}")
    print(f"Provider: {meta.provider}")
    print(f"Context: {meta.context}")
    print(f"Reasoning: {meta.supports_reasoning}")
    
    if meta.context == 256_000 and meta.supports_reasoning is True:
        print("\nSUCCESS: Metadata correctly resolved from plugin!")
    else:
        print(f"\nFAILURE: Metadata incorrect. Context: {meta.context}, Reasoning: {meta.supports_reasoning}")

except Exception as e:
    print(f"ERROR: {e}")

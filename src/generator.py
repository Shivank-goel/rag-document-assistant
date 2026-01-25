from transformers import pipeline
from typing import List 

class Generator:
    def __init__(self, model_name: str ="google/flan-t5-small"):
        self.llm = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=200)

    def generate(self, query: str, contexts: List[str]) -> str:
        context_text = "\n\n".join(contexts)

        prompt = f"""
You are an assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{query}

Answer:
"""
        result = self.llm(prompt)
        return result[0]["generated_text"]
from transformers import pipeline
from typing import List 

class Generator:
    def __init__(self, model_name: str ="google/flan-t5-small"):
        self.llm = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=300,
            do_sample=False)

    def generate(self, query: str, contexts: List[str]) -> str:
        formatted_context = "\n\n".join(
    [f"Context {i+1}: {c}" for i, c in enumerate(contexts)]
)


        prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

Rules:
- Use only the information from the context.
- If the answer is not present, say "I don't know based on the given context".
- Explain the answer in 2–4 clear sentences.
- Do not add any external knowledge.

Context:
----------------
{formatted_context}
----------------

Question:
{query}

Answer (clear and concise):
"""

        result = self.llm(prompt)
        return result[0]["generated_text"]
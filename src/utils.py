from typing import List



def clean_text(text: str) -> str:
    # Basic cleaning: strip leading/trailing whitespace and replace multiple spaces with a single space
    lines = text.splitlines()
    clean_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(clean_lines)

def chunk_text(text: str,
               chunk_size: int = 500,
               overlap: int =50
               ) -> list[str]:
    
 #Splits text into overlaping chunks
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
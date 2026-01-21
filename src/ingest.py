from pathlib import Path
from typing import Optional
from utils import clean_text, chunk_text

def load_raw_document(file_path: str) -> Optional[str]:
    try:
        #Make path relative to the project root, not current working directory 
        project_root = Path(__file__).parent.parent
        path = project_root / file_path if not Path(file_path).is_absolute() else Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()

        return text
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return None
    

    

if __name__ == "__main__":
    text = load_raw_document("data/raw/sample.txt")
    
    if text:  # Check if text loaded successfully before processing
        cleaned_text = clean_text(text)
        if cleaned_text:
            print(f"Text preview: {text[:500]}")
            chunks = chunk_text(cleaned_text)
            print(f"Total chunks: {len(chunks)}")
            if chunks:  # Check if chunks exist before accessing
                print(f" chunks: {chunks}")
            else:
                print("No chunks created.")
        else:
            print("Text cleaning failed.")
    else:
        print("Failed to load text file.")

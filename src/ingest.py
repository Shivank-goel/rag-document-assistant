from pathlib import Path
from typing import Optional

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
    if text:
        print(text[:500])
    else:
        print("No text found.")
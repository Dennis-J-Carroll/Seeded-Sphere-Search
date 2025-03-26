import os
import json
import re
from pathlib import Path

def prepare_scifi_corpus(stories_dir):
    """
    Process sci-fi stories into a corpus suitable for embedding.
    
    Args:
        stories_dir: Directory containing the sci-fi stories
        
    Returns:
        list: List of document dictionaries with id, title, and content
    """
    corpus = []
    txt_data_path = Path("/home/dennisjcarroll/Desktop/_PRO_/TXT_DATA_")
    story_files = list(txt_data_path.glob("*.txt"))
    # Get all text files in the directory
    
    for i, story_path in enumerate(story_files):
        with open(story_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract title from filename or first line
        title = story_path.stem.replace('_', ' ').title()
        
        # Clean and preprocess content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Create document entry
        doc = {
            "id": f"doc_{i}",
            "title": title,
            "content": content,
            # Extract any metadata if available
            "metadata": {"source": "scifi_collection", "path": str(story_path)}
        }
        
        corpus.append(doc)
    
    print(f"Processed {len(corpus)} sci-fi stories")
    return corpus

# Process your sci-fi stories
stories_dir = "path/to/your/scifi/stories"
corpus = prepare_scifi_corpus(stories_dir)

# Save the processed corpus
with open('scifi_corpus.json', 'w') as f:
    json.dump(corpus, f)

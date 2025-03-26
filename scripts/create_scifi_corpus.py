"""
Script to create a sci-fi corpus JSON file from text files in TXT_DATA_ directory.
This will be used for demonstrating the Seeded Sphere Search mechanism.
"""

import os
import json
import re
from tqdm import tqdm

def read_text_file(file_path):
    """Read text file and return content"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def split_into_chunks(text, max_chunk_length=1000, overlap=200):
    """Split text into overlapping chunks"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max length and we already have content,
        # save the current chunk and start a new one with overlap
        if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap by keeping the last N characters
            words = current_chunk.split()
            if len(words) > 20:  # ensure we have enough words for overlap
                current_chunk = ' '.join(words[-20:]) + ' '
            else:
                current_chunk = ""
        
        current_chunk += sentence + ' '
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def create_corpus_from_directory(directory_path, output_file="scifi_corpus.json"):
    """Process all text files in directory and create a corpus JSON file"""
    corpus = {}
    doc_id = 0
    
    # Get list of text files
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    print(f"Processing {len(files)} text files...")
    
    for file_name in tqdm(files):
        # Extract title from filename
        title = file_name.replace('_WRITING_DATA.txt', '').replace('.txt', '').replace('_', ' ')
        
        # Read file content
        file_path = os.path.join(directory_path, file_name)
        content = read_text_file(file_path)
        
        # Split into chunks
        chunks = split_into_chunks(content)
        
        # Add each chunk as a document
        for i, chunk in enumerate(chunks):
            corpus[str(doc_id)] = {
                "title": f"{title} - Part {i+1}",
                "content": chunk,
                "source": file_name,
                "chunk_index": i
            }
            doc_id += 1
    
    print(f"Created corpus with {len(corpus)} documents")
    
    # Save corpus to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"Corpus saved to {output_file}")
    return corpus

if __name__ == "__main__":
    data_dir = "TXT_DATA_"
    create_corpus_from_directory(data_dir)

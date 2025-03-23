import json
from SeededSphereSearch import SeededSphereSearch

# Load the sci-fi corpus
with open('scifi_corpus.json', 'r') as f:
    scifi_corpus = json.load(f)

# Create and initialize the Seeded Sphere Search instance
sss = SeededSphereSearch({
    "dimensions": 384,
    "min_echo_layers": 1,
    "max_echo_layers": 5,
    "frequency_scale": 0.325,
    "delta": 0.1625,
    "alpha": 0.5,
    "use_pretrained": True,
    "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2"
})

# Initialize with the sci-fi corpus
sss.initialize(scifi_corpus)

# Save the model state for future use
import pickle
with open('sss_model.pkl', 'wb') as f:
    pickle.dump(sss, f)

# Test a simple search
results = sss.search("space exploration with alien technology", top_k=5)
for i, result in enumerate(results):
    print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
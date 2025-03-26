# Seeded Sphere Search Tester UI

This is a web-based interface for testing the Seeded Sphere Search functionality. It allows you to:

- Perform semantic searches with the Seeded Sphere Search engine
- Use seed documents to influence search results
- Apply transformations (like Ellipsoidal Transformation) to the search space
- Configure search parameters like threshold and max results
- View search results with similarity scores and metadata

## Requirements

- Python 3.6+
- Flask
- Flask-CORS
- A pre-trained Seeded Sphere Search model file (`sss_model.pkl`)

## Installation

Make sure you have the required Python packages installed:

```bash
pip install flask flask-cors
```

## Usage

1. Navigate to this directory
2. Run the server:

```bash
python seeded_sphere_search_tester.py /path/to/your/sss_model.pkl
```

If you don't specify a model path, it will look for `sss_model.pkl` in the current directory.

3. Open your browser and go to `http://localhost:5000`

## Features

### Basic Search

- Enter a search query in the search box
- Click "Search" or press Enter to perform a search
- View the results with similarity scores

### Seeded Search

- Select a seed document from the dropdown menu
- Enter a search query
- The search results will be influenced by the seed document's embedding
- This is useful for finding documents that are similar to both the query and the seed

### Advanced Options

- **Similarity Threshold**: Filter results by minimum similarity score (0-1)
- **Max Results**: Limit the number of results returned (1-50)
- **Ellipsoidal Transformation**: Apply ellipsoidal transformation to embeddings (if available)

## How It Works

1. The UI loads the pre-trained Seeded Sphere Search model
2. When you perform a search:
   - Without a seed: It uses the standard search functionality
   - With a seed: It combines the query embedding with the seed document's embedding
3. It then ranks documents by similarity and displays the results

## Customization

You can modify the `seeded_sphere_search_tester.py` file to change:

- The weighting between query and seed (currently 50/50)
- The UI appearance and behavior
- The search algorithm parameters

## Troubleshooting

- If you see "Search engine not loaded" error, make sure your model file path is correct
- If ellipsoidal transformation is not available, check that you have `ellipsoidal_weights.npy` file 
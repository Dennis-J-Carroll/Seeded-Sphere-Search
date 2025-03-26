from sklearn.feature_extraction.text import TfidfVectorizer

# Test the vectorizer
docs = ["This is a test", "Another test document"]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)
print("TfidfVectorizer works!")
print(f"Matrix shape: {tfidf_matrix.shape}")
print(f"Feature names: {vectorizer.get_feature_names_out()}")

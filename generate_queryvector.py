import numpy as np
from sklearn.preprocessing import normalize

# Vector dimensions and count
D = 128  # Embedding size
QUERY_VECTOR_COUNT = 1000  # Number of query vectors
QUERY_VECTOR_FILE = "query_vectors.txt"

# Generate random normalized query vectors
def generate_query_vectors(n, d):
    vectors = np.random.rand(n, d).astype(np.float32)
    return normalize(vectors)  # Normalize for cosine similarity

# Generate and save query vectors
query_vectors = generate_query_vectors(QUERY_VECTOR_COUNT, D)
np.savetxt(QUERY_VECTOR_FILE, query_vectors, delimiter=",")

print(f"Generated {QUERY_VECTOR_COUNT} query vectors and saved to {QUERY_VECTOR_FILE}")

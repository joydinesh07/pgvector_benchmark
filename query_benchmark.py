import psycopg2
import numpy as np
import time
import concurrent.futures

# PostgreSQL connection parameters
DB_CONFIG = {
    "dbname": "your_db",
   # "user": "your_user",
   # "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# Load query vectors from file
def load_query_vectors(file_path):
    return np.loadtxt(file_path, delimiter=",", dtype=np.float32)

# Function to execute a query
def execute_query(query_vector):
    query = """
        SELECT id, embedding <-> %s::vector AS distance
        FROM embeddings
        ORDER BY distance LIMIT 10;
    """
    start_time = time.time()
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (query_vector.tolist(),))  # Make sure the vector is in list format
            results = cursor.fetchall()
    return time.time() - start_time

# Execute queries concurrently
def benchmark_queries(file_path, num_concurrent_queries=100):
    query_vectors = load_query_vectors(file_path)
    query_times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_queries) as executor:
        results = list(executor.map(execute_query, query_vectors))
        query_times.extend(results)
    
    # Compute percentiles
    p90 = np.percentile(query_times, 90)
    p95 = np.percentile(query_times, 95)
    p99 = np.percentile(query_times, 99)
    
    print(f"Query Response Times:")
    print(f"90th percentile: {p90:.4f} seconds")
    print(f"95th percentile: {p95:.4f} seconds")
    print(f"99th percentile: {p99:.4f} seconds")

# Run benchmark (Provide the correct file path)
benchmark_queries("query_vectors.txt")


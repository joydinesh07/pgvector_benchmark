import psycopg2
import numpy as np
import time
import concurrent.futures
import random
from sklearn.preprocessing import normalize

# PostgreSQL connection parameters
DB_CONFIG = {
    "dbname": "your_db",
    #"user": "your_user",
    #"password": "your_password",
    "host": "localhost",
    "port": "5432"
}

QUERY_VECTOR_FILE = "query_vectors.txt"
NUM_CONCURRENT_USERS = 100  # Number of concurrent queries
TEST_DURATION = 60  # Test duration in seconds (5 minutes)

# Load query vectors from file
def load_query_vectors(file_path):
    return np.loadtxt(file_path, delimiter=",", dtype=np.float32)

query_vectors = load_query_vectors(QUERY_VECTOR_FILE)

# Function to execute a single query
def execute_query(query_vector):
    query = """
        SELECT id, embedding <-> %s::vector AS distance
        FROM embeddings
        ORDER BY distance LIMIT 10;
    """
    start_time = time.time()
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (query_vector.tolist(),))
            cursor.fetchall()
    return time.time() - start_time  # Return query execution time

# Benchmark function
def benchmark_queries():
    start_time = time.time()
    query_count = 0
    query_times = []

    def worker():
        nonlocal query_count
        while time.time() - start_time < TEST_DURATION:
            random_vector = random.choice(query_vectors)  # Pick random query vector
            elapsed_time = execute_query(random_vector)
            query_times.append(elapsed_time)
            query_count += 1

    # Run concurrent queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_USERS) as executor:
        futures = [executor.submit(worker) for _ in range(NUM_CONCURRENT_USERS)]

        # Print QPS in real-time
        while time.time() - start_time < TEST_DURATION:
            elapsed_time = time.time() - start_time
            qps = query_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Elapsed: {elapsed_time:.2f}s | Queries: {query_count} | QPS: {qps:.2f}")
            time.sleep(1)

    total_time = time.time() - start_time
    avg_qps = query_count / total_time

    # Compute response time percentiles
    p90 = np.percentile(query_times, 90)
    p95 = np.percentile(query_times, 95)
    p99 = np.percentile(query_times, 99)

    print(f"\n=== Benchmark Completed ===")
    print(f"Total Queries: {query_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average QPS: {avg_qps:.2f}")
    print(f"90th percentile response time: {p90:.4f} sec")
    print(f"95th percentile response time: {p95:.4f} sec")
    print(f"99th percentile response time: {p99:.4f} sec")

# Run benchmark
benchmark_queries()

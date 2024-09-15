from flask import Flask, request, jsonify
import redis
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2

app = Flask(__name__)

# Redis caching
cache = redis.Redis(host='localhost', port=6379, db=0)

# Database connection
db_conn = psycopg2.connect(
    host="localhost",
    database="ML",
    user="postgres",
    password="postgres"
)

# Load a sentence transformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Rate limit setup: Max 5 requests per user
USER_REQUEST_LIMIT = 5

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is active"}), 200

# Function to get embeddings for search
def get_embedding(text):
    return model.encode([text])[0]

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    user_id = data.get("user_id")
    query = data.get("text", "")
    top_k = int(data.get("top_k", 5))
    threshold = float(data.get("threshold", 0.5))

    # Check user request count in Redis
    user_requests = cache.get(f"user:{user_id}:requests")
    
    if user_requests and int(user_requests) >= USER_REQUEST_LIMIT:
        return jsonify({"error": "Rate limit exceeded"}), 429

    # Increment request count
    cache.incr(f"user:{user_id}:requests")
    
    # Cache request timeout reset
    cache.expire(f"user:{user_id}:requests", 3600)  # Reset after an hour

    start_time = time.time()

    query_embedding = get_embedding(query)

    # Fetch documents from the database
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, content, embedding FROM documents")
    rows = cursor.fetchall()

    documents = []
    for row in rows:
        doc_id, content, embedding = row
        embedding = np.frombuffer(embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        if similarity >= threshold:
            documents.append((doc_id, content, similarity))

    # Sort by similarity and limit to top_k
    documents = sorted(documents, key=lambda x: x[2], reverse=True)[:top_k]
    end_time = time.time()

    inference_time = end_time - start_time

    return jsonify({
        "results": [{"id": doc[0], "content": doc[1], "similarity": doc[2]} for doc in documents],
        "inference_time": inference_time
    })

if __name__ == '__main__':
    app.run(debug=True)

from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample documents
documents = [
    "Artificial intelligence is transforming industries.",
    "Machine learning algorithms learn from data.",
    "Natural language processing helps computers understand text.",
    "Flask is a web framework for building applications.",
    "PostgreSQL is a powerful, open-source database."
]

# Connect to PostgreSQL
db_conn = psycopg2.connect(
    host="localhost",
    database="document_db",
    user="your_user",
    password="your_password"
)

cursor = db_conn.cursor()

for doc in documents:
    embedding = model.encode([doc])[0].tobytes()
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (doc, embedding))

db_conn.commit()
cursor.close()
db_conn.close()

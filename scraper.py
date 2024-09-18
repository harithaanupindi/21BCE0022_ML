import requests
from bs4 import BeautifulSoup
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="document_db",
    user="postgres",
    password="postgres",
)
cursor = conn.cursor()

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def fetch_articles():
    # Example URL (replace with actual news sources)
    url = 'https://news.ycombinator.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Example scraping (this needs to be tailored to the actual HTML structure)
    articles = soup.find_all('a', class_='storylink')
    for article in articles:
        title = article.get_text()
        content = title  # For simplicity, content is the same as title
        
        # Compute embedding
        embedding = model.encode([content])[0]
        embedding = embedding.astype(np.float32).tobytes()

        # Insert into database
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding)
        )

fetch_articles()
conn.commit()
cursor.close()
conn.close()

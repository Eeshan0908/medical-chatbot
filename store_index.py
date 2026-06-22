from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS

# Load data
documents = load_pdf_file("data")   # make sure your PDFs are inside "data" folder

# Split text
text_chunks = text_split(documents)

# Get embeddings
embeddings = download_hugging_face_embeddings()

# Create FAISS index
db = FAISS.from_documents(text_chunks, embeddings)

# Save index locally
db.save_local("faiss_index")

print("FAISS index created successfully!")
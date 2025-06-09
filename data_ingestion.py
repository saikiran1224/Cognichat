import os
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings
from pymongo import MongoClient
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader

import config # Importing config module for constants

# Load environment variables from .env file
load_dotenv()

# --- Configuration for MongoDB and Embeddings ---

# Initialize Ollama Embeddings - (IMP: SHOULD BE THE SAME EMBEDDING MODEL USED INITIALLY DURING DATA INGESTION)
embeddings = OllamaEmbeddings(model="llama3.2")

# Initializing MongoDB python client
try:
    client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI")) # fetching MongoDB Atlas URI from environment variable
    MONGODB_COLLECTION = client[config.DB_NAME][config.COLLECTION_NAME]

    print(f"Connected to MongoDB Atlas: {config.DB_NAME}.{config.COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Initialize MongoDBAtlasVectorSearch (this will connect to the existing collection)
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings, # Using Ollama embeddings to embed the chunks during ingestion to Vector store
    collection=MONGODB_COLLECTION, # Collection where the embeddings will be stored
    index_name=config.ATLAS_VECTOR_SEARCH_INDEX_NAME, # Name of the vector search index
    relevance_score_fn="cosine", # Using cosine similarity for relevance scoring
)

# --- Create Vector Index for MongoDB Atlas Vector Search which is required for vector search ---
# This step is crucial as it sets up the index for vector search capabilities.

print(f"Attempting to create vector search index: {config.ATLAS_VECTOR_SEARCH_INDEX_NAME}...")

# As LLama 3.2 embeddings are 3072 dimensions, we need to ensure the index is created with the correct dimensions so there will be no error during search operation.
try:
    vector_store.create_vector_search_index(dimensions=3072)
    print("Vector search index created successfully (or already exists).")

except Exception as e:
    print(f"Error creating vector search index: {e}")
    # Continue if the error is due to index already existing, otherwise handle.

# --- Data Loading and Chunking ---
# Case 1: Loading from Website
print("Loading documents...")
website_loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = website_loader.load()
print(f"Loaded {len(docs)} documents.")

# # Case 2: Loading from Local Files (Uncomment if needed)
# loader = DirectoryLoader(
#         "data/website_data",
#         glob="**/*.md",
#         show_progress=True,
#         recursive=True,
#         loader_cls=UnstructuredMarkdownLoader,
# )
# # Load documents from PDF, PPT, Excel, and Word files
# pdf_loader = UnstructuredPDFLoader("data/documents/*.pdf")
# ppt_loader = UnstructuredPPTXLoader("data/documents/*.pptx")
# excel_loader = UnstructuredExcelLoader("data/documents/*.xlsx")
# word_loader = UnstructuredWordLoader("data/documents/*.docx")

# # Load documents from each type
# pdf_docs = pdf_loader.load()
# ppt_docs = ppt_loader.load()
# excel_docs = excel_loader.load()
# word_docs = word_loader.load()

# # Combine all loaded documents
# docs.extend(pdf_docs)
# docs.extend(ppt_docs)
# docs.extend(excel_docs)
# docs.extend(word_docs)

# print(f"Loaded {len(pdf_docs)} PDF documents, {len(ppt_docs)} PPT documents, {len(excel_docs)} Excel documents, and {len(word_docs)} Word documents.")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"Created {len(all_splits)} chunks.")

# --- Storing Embeddings in MongoDB Atlas Vector Search ---
print("Adding document chunks to MongoDB Atlas Vector Search...")
try:
    _ = vector_store.add_documents(documents=all_splits)
    print("Document chunks successfully embedded and stored in MongoDB Atlas.")
except Exception as e:
    print(f"Error adding documents to vector store: {e}")

print("Ingestion process complete.")
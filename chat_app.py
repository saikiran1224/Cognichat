import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain import hub
from langchain_core.documents import Document
from typing import List, TypedDict
from langgraph.graph import START, StateGraph

import config

# Load environment variables from .env file
load_dotenv()

# --- Configuration for LLM, Embeddings, and MongoDB ---
# Initialize your LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
print("LLM (Gemini 2.0 Flash) initialized.")

# Initialize Ollama Embeddings (ensure Ollama server is running with llama3.2)
# IMPORTANT: This must be the SAME embedding model used during ingestion.
embeddings = OllamaEmbeddings(model="llama3.2")
print("Ollama Embeddings (llama3.2) initialized.")

# MongoDB Connection details (must match ingestion.py)

# Initialize MongoDB python client
try:
    client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
    MONGODB_COLLECTION = client[config.DB_NAME][config.COLLECTION_NAME]
    print(f"Connected to MongoDB Atlas: {config.DB_NAME}.{config.COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Initialize MongoDBAtlasVectorSearch using the existing collection and index
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=config.ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)
print("MongoDB Atlas Vector Store initialized with existing embeddings.")

# --- Define prompt for question-answering ---
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")
print("RAG prompt pulled from LangChain Hub.")

# --- Define state for application ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# --- Graph Nodes ---
# This method retrieves relevant documents from the vector store based on the user's question
def retrieve(state: State):
    print(f"\nRetrieving documents for question: '{state['question']}'...")
    retrieved_docs = vector_store.similarity_search(state["question"])
    print(f"Retrieved {len(retrieved_docs)} documents.")
    return {"context": retrieved_docs}

# This method generates an answer to the user's question using the retrieved context
def generate(state: State):
    print("Generating answer based on retrieved context...")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    print("Answer generated.")
    return {"answer": response.content}

# --- Build LangGraph ---
print("Building LangGraph...")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
print("LangGraph compiled.")

# --- Chat Application Loop ---
print("\n--- RAG Chat Application Started ---")
print("Type 'exit' to quit.")

while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == 'exit':
        print("Exiting chat application. Goodbye!")
        break

    try:
        response = graph.invoke({"question": user_question})
        print("\nAnswer:")
        print(response["answer"])
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try again.")
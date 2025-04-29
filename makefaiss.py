import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY")  # Corrected variable name
aws_secret_access_key = os.getenv("AWS_SECRET_KEY")  # Corrected variable name
aws_region = os.getenv("AWS_REGION")



# Bedrock model ID (Llama 3.1 70B Instruct)
# model_id = "meta.llama3-70b-instruct-v1:0"

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_runtime)

# ---- PDF Loading and Chunking ----
def load_and_chunk_pdfs(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    return docs

# ---- Build and Save FAISS Vector Store ----
def build_and_save_faiss_index(folder_path, faiss_dir="faiss_index"):
    docs = load_and_chunk_pdfs(folder_path)
    vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore.save_local(faiss_dir)
    print(f"✅ FAISS index saved at: {faiss_dir}")

# ---- Run ----
if __name__ == "__main__":
    PDF_FOLDER = "data"  # Path to your PDF folder
    FAISS_DIR = "faiss_index"
    build_and_save_faiss_index(PDF_FOLDER, FAISS_DIR)

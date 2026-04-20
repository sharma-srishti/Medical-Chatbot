# from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import torch
import os
load_dotenv()

# LLM setup
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)



# Extract text from pdf file
def load_pdf(file_path):
    loader = DirectoryLoader(file_path,glob="*.pdf",loader_cls=PyPDFLoader) # glob make sure I ean to load pdf only
    documents = loader.load()
    return documents

# Filter the documents to keep only 'source' in metadata and the original page_content
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
        Given a list of Document objects, return a new list of Document objects 
        containing only 'source' in metadata and the original page_content.
    """

    minimal_docs :List[Document]= []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": src}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

# Split the page_content of each Document into chunks 
def text_splitter(minimal_docs):
    """
        Given a list of Document objects, split the page_content into chunks of specified size and overlap.
        Return a new list of Document objects with the split content and original metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500, 
                            chunk_overlap=20)
    
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks
    

# Get the HuggingFace BGE embeddings model
def download_embeddings():
    """
        Download the HuggingFace BGE embeddings model and return an instance of HuggingFaceBgeEmbeddings.
    """
    model_name = "BAAI/bge-small-en-v1.5"

    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return embeddings


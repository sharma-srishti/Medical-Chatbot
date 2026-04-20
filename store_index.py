import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf,filter_to_minimal_docs, text_splitter,download_embeddings
load_dotenv()


extracted_data = load_pdf("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter(minimal_docs)
embeddings = download_embeddings()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    ) 
index=pc.Index(index_name)

# Load existing index or create a new one
from langchain_pinecone import PineconeVectorStore
doc_search = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,   
    index_name=index_name
)

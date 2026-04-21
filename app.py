from flask import Flask,render_template,jsonify,request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
from src.helper import download_embeddings
import os
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = download_embeddings()

index_name = "medical-bot"
doc_search = PineconeVectorStore.from_existing_index(
    embedding=embedding,   
    index_name=index_name
)

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "{input}"}
])

memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[user_id]

# question_answering_chain = create_stuff_documents_chain(model,prompt)
# rag_chain = create_retrieval_chain(retriever, question_answering_chain)

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    user_input = request.form.get("msg")
    user_id = request.form.get("user_id")

    if not user_id:
        return "user_id missing"

    # ✅ Get memory for this user
    memory = get_memory(user_id)

    # ✅ Create conversational chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    # ✅ Invoke
    response = rag_chain.invoke({
        "question": user_input
    })

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8000,debug=True)

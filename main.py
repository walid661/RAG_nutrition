import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, WebBaseLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Charger la clé API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY manquante dans le fichier .env")

# Initialisation de FastAPI
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Charger les documents depuis le dossier docs/
docs = []
docs_folder = "docs"

if os.path.exists(docs_folder):
    for filename in os.listdir(docs_folder):
        filepath = os.path.join(docs_folder, filename)

        try:
            if filename.endswith(".txt"):
                docs.extend(TextLoader(filepath).load())

            elif filename.endswith(".pdf"):
                docs.extend(PyPDFLoader(filepath).load())

            elif filename.endswith(".docx"):
                docs.extend(UnstructuredWordDocumentLoader(filepath).load())

            elif filename.endswith(".urls"):
                with open(filepath, "r", encoding="utf-8") as f:
                    urls = f.readlines()
                for url in urls:
                    if url.strip():
                        docs.extend(WebBaseLoader(url.strip()).load())

        except Exception as e:
            print(f"Erreur lors du chargement de {filename} : {e}")
else:
    print("Dossier docs/ introuvable. La base sera vide.")

# 2. Découpage en chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# 3. Embeddings et indexation
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

# 4. Modèle LLM et chaîne RAG
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Modèle pour la requête /ask
class AskRequest(BaseModel):
    query: str

# Endpoint de test
@app.get("/")
def home():
    return {"message": "RAG backend fonctionne"}

# Endpoint principal
@app.post("/ask")
def ask_rag(request: AskRequest):
    try:
        result = qa.run(request.query)
        return {"question": request.query, "answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

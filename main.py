import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 1. Charger les variables d'environnement
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Initialiser FastAPI
app = FastAPI()

# 3. Autoriser CORS (utile pour ton front HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Préparer les embeddings et modèles
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# 5. Charger les documents depuis docs/
docs = []
docs_dir = "docs"

if os.path.exists(docs_dir):
    for file in os.listdir(docs_dir):
        path = os.path.join(docs_dir, file)
        try:
            if file.endswith(".txt"):
                docs.extend(TextLoader(path).load())
            elif file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"Erreur lors du chargement de {file} : {e}")

# 6. Construire la base vectorielle (si docs présents)
if docs:
    db = Chroma.from_documents(docs, embeddings)
else:
    print("Aucun document chargé, la base est vide.")
    db = None

# 7. Définir le schéma de requête
class AskRequest(BaseModel):
    query: str

# 8. Route test
@app.get("/")
def home():
    return {"message": "Backend RAG Nutrition fonctionne"}

# 9. Route principale
@app.post("/ask")
def ask(request: AskRequest):
    if not db:
        return {"response": "⚠️ La base est vide. Ajoutez des fichiers dans docs/."}
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain.run(request.query)
    return {"response": result}


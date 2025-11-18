import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Charger les variables d'environnement du fichier .env local
load_dotenv()

DOCUMENTS_PATH = "documents"
DB_PATH = "chroma_db"

def main():
    """
    Fonction principale pour créer la base de données vectorielle en local.
    """
    print("Début de l'ingestion des documents locaux...")

    # Recommandé : supprimer l'ancienne base pour éviter les conflits
    if os.path.exists(DB_PATH):
        print(f"Suppression de l'ancienne base '{DB_PATH}'...")
        shutil.rmtree(DB_PATH)

    if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
        print(f"Erreur : Le dossier '{DOCUMENTS_PATH}' est vide ou n'existe pas.")
        return

    # 1. Charger les documents
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    print(f"{len(documents)} document(s) chargé(s).")

    # 2. Découper les documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"✂️  Les documents ont été découpés en {len(texts)} morceaux (chunks).")

    # 3. Créer les embeddings et stocker dans ChromaDB
    print("Création des embeddings et stockage dans ChromaDB... (cela peut prendre un moment)")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # La création et la persistance se font en une seule étape
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)

    print("-" * 50)
    print(f"✅ L'ingestion est terminée.")
    print(f"La base de données vectorielle a été créée dans le dossier '{DB_PATH}'.")
    print("-" * 50)


if __name__ == "__main__":
    main()

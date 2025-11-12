import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Définir les chemins
DOCUMENTS_PATH = "documents"
DB_PATH = "chroma_db"


def create_vector_db():
    """
    Cette fonction crée une base de données vectorielle à partir des documents PDF.
    """
    print("Début de la création de la base de données vectorielle...")

    # 1. Charger les documents PDF du dossier spécifié
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("Aucun document PDF trouvé. Veuillez ajouter des fichiers PDF dans le dossier 'documents'.")
        return

    print(f"{len(documents)} document(s) chargé(s).")

    # 2. Découper les documents en morceaux (chunks)
    # On découpe les textes pour qu'ils ne soient pas trop longs pour le modèle.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} morceaux de texte créés.")

    # 3. Créer les embeddings (vecteurs numériques)
    # On utilise le modèle d'embedding d'OpenAI.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Créer et persister la base de données vectorielle ChromaDB
    # Les textes et leurs embeddings sont stockés dans le dossier spécifié (DB_PATH).
    print("Création et stockage des embeddings dans ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)

    print("-" * 50)
    print(f"Base de données vectorielle créée avec succès dans le dossier '{DB_PATH}' !")
    print(f"Nombre de chunks vectorisés et stockés : {db._collection.count()}")
    print("-" * 50)


if __name__ == "__main__":
    create_vector_db()

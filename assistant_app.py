import streamlit as st
import datetime
import os
import time

# --- Imports Logiques ---
from dotenv import load_dotenv

# RAG
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Agents et Outils
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.chains import LLMMathChain
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Mémoire
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

# --- FONCTIONS PRINCIPALES ---

def ingest_documents():
    """Charge, découpe et vectorise les documents PDF."""
    DOCUMENTS_PATH = "documents"
    DB_PATH = "chroma_db"

    if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
        st.warning("Le dossier 'documents' est vide. L'outil de recherche interne ne sera pas disponible.", icon="⚠️")
        return False

    with st.status("Initialisation de l'outil RAG : ingestion des documents...", expanded=True) as status:
        st.write("Chargement des documents PDF...")
        loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        time.sleep(1)

        st.write("Découpage des textes...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        time.sleep(1)

        st.write("Vectorisation et stockage dans ChromaDB...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
        db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
        time.sleep(1)

        status.update(label="Outil RAG prêt à l'emploi !", state="complete", expanded=False)
    return True


# --- CONFIGURATION DE L'APPLICATION ---
st.set_page_config(page_title="Assistant Intelligent Multi-Compétences", page_icon="🤖")
st.title("🤖 Assistant Intelligent Multi-Compétences")
st.caption("Je peux répondre à des questions sur vos documents, chercher sur le web, calculer, et plus encore.")

# Vérification et ingestion des documents au démarrage de l'application
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = ingest_documents()

# --- CONFIGURATION DES OUTILS ET DE L'AGENT ---

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])


def get_current_datetime(user_input: str = "") -> str:
    """Renvoie la date et l'heure actuelles dans un format lisible."""
    now = datetime.datetime.now()
    return f"La date et l'heure actuelles sont : {now.strftime('%A %d %B %Y, %H:%M:%S')}."


def setup_rag_tool():
    """Crée l’outil RAG connecté à la base Chroma persistée."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["OPENAI_API_KEY"])
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return Tool(
        name="Recherche Documents Internes",
        func=rag_chain.invoke,
        description=(
            "INDISPENSABLE pour répondre aux questions sur les documents internes, "
            "les politiques de l'entreprise, les manuels et les rapports spécifiques. "
            "Utilisez ceci pour toute question concernant le contenu des fichiers PDF fournis."
        )
    )


tools = []
if st.session_state.rag_initialized:
    tools.append(setup_rag_tool())

# Ajout des autres outils
tools.extend([
    Tool(
        name="Recherche Web",
        func=DuckDuckGoSearchRun().run,
        description="Utile pour rechercher sur internet des informations très récentes ou générales."
    ),
    Tool(
        name="Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Utile pour rechercher des informations factuelles sur des sujets encyclopédiques."
    ),
    Tool(
        name="Calculatrice",
        func=LLMMathChain.from_llm(llm=llm).run,
        description="Utile pour effectuer des calculs mathématiques."
    ),
    Tool(
        name="Horloge",
        func=get_current_datetime,
        description="Utile pour obtenir la date et l'heure actuelles."
    )
])

# --- CONFIGURATION DE L'AGENT AVEC MÉMOIRE ---
prompt = hub.pull("hwchase17/react-chat")
prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# --- INTERFACE STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            response = agent_executor.invoke({"input": user_query})
            st.markdown(response["output"])

    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

import streamlit as st
import datetime
import os
import time
from dotenv import load_dotenv

# --- Imports Logiques ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub  # ✅ bon import
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.chains import LLMMathChain
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

# --- GESTION DE LA CLÉ API (Local vs. Déploiement) ---
try:
    # Essayer de lire les secrets de Streamlit (pour le déploiement)
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    # Si ça échoue, on est en local, donc on charge le .env
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- FONCTION PRINCIPALE D’INGESTION ---
def ingest_documents():
    DOCUMENTS_PATH = "documents"
    DB_PATH = "chroma_db"

    if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
        st.warning("⚠️ Le dossier 'documents' est vide. L'outil de recherche interne ne sera pas disponible.")
        return False

    with st.status("📚 Ingestion des documents en cours...", expanded=True) as status:
        st.write("📥 Chargement des documents PDF...")
        loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        time.sleep(1)

        st.write("✂️ Découpage des textes...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        time.sleep(1)

        st.write("🧠 Vectorisation et stockage dans ChromaDB...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
        time.sleep(1)

        status.update(label="✅ Outil RAG prêt à l'emploi !", state="complete", expanded=False)
    return True

# --- CONFIGURATION DE L'APPLICATION ---
st.set_page_config(page_title="Assistant Intelligent Multi-Compétences", page_icon="🤖")
st.title("🤖 Assistant Intelligent Multi-Compétences")
st.caption("Je peux répondre à des questions sur vos documents, chercher sur le web, calculer, et plus encore.")

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = ingest_documents()

# --- CONFIGURATION DES OUTILS ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

def get_current_datetime(_: str = "") -> str:
    now = datetime.datetime.now()
    return f"🕒 Nous sommes le {now.strftime('%A %d %B %Y, %H:%M:%S')}."

def setup_rag_tool():
    """Outil RAG basé sur les documents locaux."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever()

    template = """Répondez à la question en vous basant uniquement sur le contexte suivant :
    {context}

    Question : {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return Tool(
        name="Recherche Documents Internes",
        func=rag_chain.invoke,
        description=(
            "Indispensable pour répondre aux questions sur les documents internes, "
            "les politiques de l'entreprise, les manuels et les rapports PDF."
        )
    )

tools = []
if st.session_state.get("rag_initialized", False):
    tools.append(setup_rag_tool())

tools.extend([
    Tool(
        name="Recherche Web",
        func=DuckDuckGoSearchRun().run,
        description="Utile pour rechercher sur internet des informations récentes ou générales."
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
    ),
])

# --- CONFIGURATION DE L'AGENT AVEC MÉMOIRE ---
prompt = hub.pull("hwchase17/react-chat")  # ✅ modèle correct disponible sur LangChain Hub
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

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
        with st.spinner("🤔 Réflexion..."):
            response = agent_executor.invoke({"input": user_query})
            st.markdown(response["output"])

    st.session_state.messages.append({"role": "assistant", "content": response["output"]})

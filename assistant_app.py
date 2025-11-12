import streamlit as st
from dotenv import load_dotenv

# --- Imports LangChain ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA, LLMMathChain
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- Charger les variables d'environnement ---
load_dotenv()

# --- CONFIGURATION ---
DB_PATH = "chroma_db"

# --- Initialisation du modèle principal ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# --- Outil RAG : recherche dans la base Chroma ---
def setup_rag_tool():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    return Tool(
        name="Recherche Documents Internes",
        func=lambda q: rag_chain.invoke({"query": q}),
        description="Permet de répondre aux questions sur les documents PDF internes ou les bases Chroma."
    )

# --- Autres outils ---
search = DuckDuckGoSearchRun()
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
llm_math_chain = LLMMathChain.from_llm(llm=llm)

tools = [
    setup_rag_tool(),
    Tool(
        name="Recherche Web",
        func=search.run,
        description="Recherche d'informations récentes ou générales sur Internet."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Recherche d'informations factuelles sur des sujets encyclopédiques."
    ),
    Tool(
        name="Calculatrice",
        func=llm_math_chain.run,
        description="Effectue des calculs mathématiques simples."
    )
]

# --- Création de l'agent ---
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# --- Exécuteur de l'agent ---
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- Interface Streamlit ---
st.set_page_config(page_title="Assistant Intelligent Multi-Compétences", page_icon="🤖")
st.title("🤖 Assistant Intelligent Multi-Compétences")
st.caption("Posez-moi des questions sur vos documents, le web, ou effectuez des calculs.")

# --- Historique de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if user_query := st.chat_input("Posez votre question ici..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                answer = response["output"]
            except Exception as e:
                answer = f"⚠️ Une erreur est survenue : {e}"

            st.markdown(answer)

    # Sauvegarde de la réponse
    st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Charger les variables d'environnement
load_dotenv()

# Définir le chemin de la base de données
DB_PATH = "chroma_db"


def format_docs(docs):
    """Helper function to format documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # --- Configuration de la page Streamlit ---
    st.set_page_config(page_title="Assistant de Documents", page_icon="📄")
    st.title("📄 Assistant Intelligent (RAG)")
    st.caption("Posez des questions sur le contenu de vos documents PDF.")

    # --- Chargement de la base de données vectorielle ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # --- Création de la chaîne de traitement (style LCEL) ---
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template de prompt pour guider le LLM
    template = """
    Vous êtes un assistant spécialisé dans la réponse aux questions basées sur des documents.
    Utilisez les morceaux de contexte suivants pour répondre à la question à la fin.
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
    Gardez la réponse concise et informative.

    Contexte:
    {context}

    Question:
    {question}

    Réponse utile:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Création de la chaîne LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Interface utilisateur ---
    query = st.text_input(
        "Posez votre question ici :",
        placeholder="Ex : Quel est le sujet principal du rapport X ?"
    )

    if query:
        with st.spinner("Recherche de la réponse dans les documents..."):
            # On exécute la chaîne avec la question de l'utilisateur
            answer = rag_chain.invoke(query)

            # Affichage de la réponse
            st.header("Réponse")
            st.write(answer)

            # Affichage des sources (retrouvées par le retriever)
            with st.expander("Afficher les sources utilisées"):
                source_documents = retriever.invoke(query)
                for document in source_documents:
                    st.info(
                        f"Source : {document.metadata['source']} "
                        f"(Page : {document.metadata.get('page', 'N/A')})"
                    )
                    st.write(document.page_content)


if __name__ == "__main__":
    main()

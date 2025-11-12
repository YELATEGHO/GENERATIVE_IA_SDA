import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.chains import LLMMathChain
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# Charger les variables d'environnement (.env)
load_dotenv()


def test_agent():
    """
    Agent ReAct compatible LangChain 0.3.x
    """
    print("Initialisation de l'agent (LangChain 0.3.13)...")

    # Initialisation du modèle OpenAI
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # --- Outils disponibles ---
    search = DuckDuckGoSearchRun()
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    llm_math_chain = LLMMathChain.from_llm(llm=llm)

    tools = [
        Tool(
            name="Recherche Web",
            func=search.run,
            description="Utile pour rechercher sur internet des informations récentes ou générales.",
        ),
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Utile pour rechercher des informations factuelles sur des sujets encyclopédiques.",
        ),
        Tool(
            name="Calculatrice",
            func=llm_math_chain.run,
            description="Utile pour effectuer des calculs mathématiques simples.",
        ),
    ]

    # --- Charger le prompt ReAct depuis LangChain Hub ---
    prompt = hub.pull("hwchase17/react")

    # --- Créer l’agent ---
    agent = create_react_agent(llm, tools, prompt)

    # --- Créer l’exécuteur ---
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("✅ Agent prêt. Tapez 'quitter' pour arrêter.\n")

    # --- Boucle interactive ---
    while True:
        try:
            query = input("Votre question : ")
            if query.lower() == "quitter":
                break

            result = agent_executor.invoke({"input": query})
            print("\n🧠 Réponse finale :", result["output"], "\n")

        except Exception as e:
            print(f"❌ Une erreur est survenue : {e}\n")


if __name__ == "__main__":
    test_agent()

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- 1. Load Documents ---
print("Loading documents...")
text_loader_kwargs={'autodetect_encoding': True}
pdf_loader = DirectoryLoader("./data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
text_loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
documents = pdf_loader.load() + text_loader.load()
print(f"Loaded {len(documents)} documents.")

# --- 2. Split Documents ---
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# --- 3. Create Embeddings & Store in ChromaDB ---
print("Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
retriever = vectorstore.as_retriever()
print("Embeddings created and stored.")

# --- 4. Define Tools for the Agent ---
print("Defining tools for the agent...")

# Tool 1: A retriever tool for searching local documents
retriever_tool = create_retriever_tool(
    retriever,
    "local_document_search",
    "Searches and returns information from the local document collection. Use this for any questions about the content of the provided documents.",
)

# Tool 2: A tool for performing web searches using Tavily
search_tool = TavilySearchResults()

# Combine the tools into a list for the agent
tools = [retriever_tool, search_tool]
print("Tools defined.")

# --- 5. Create the Agent ---
print("Creating the agent...")
# Initialize the Gemini 1.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# Get a prompt template that is designed for tool-calling agents
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create the tool-calling agent. This is a more modern and robust way to build agents.
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("Agent created.")


# --- 6. Start the Chatbot ---
print("\n--- Agentic Chatbot is ready! ---")
print("Ask a question or type 'exit' to quit.")

chat_history = []

while True:
    query = input("\nYou: ")
    if query.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    # Invoke the agent executor with the user's query and chat history
    try:
        result = agent_executor.invoke({"input": query, "chat_history": chat_history})
        
        # Print the agent's response
        print(f"\nBot: {result['output']}")

        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["output"]))
    except Exception as e:
        print(f"An error occurred: {e}")


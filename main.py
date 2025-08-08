import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever


# Load environment variables from .env file
load_dotenv()

# --- 1. Load Documents ---
# Load documents from the 'data' directory. It will load .txt and .pdf files.
print("Loading documents...")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader("./data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
text_loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

documents = loader.load()
text_documents = text_loader.load()
documents.extend(text_documents)
print(f"Loaded {len(documents)} documents.")


# --- 2. Split Documents ---
# Split the documents into smaller chunks for better processing.
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")


# --- 3. Create Embeddings & Store in ChromaDB ---
# Create embeddings for the document chunks and store them in a Chroma vector store.
# This will create a 'chroma_db' directory to persist the database.
print("Creating embeddings and storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
print("Embeddings created and stored.")


# --- 4. Set up the Conversational RAG Chain ---
print("Setting up the conversational RAG chain...")
# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# The retriever to fetch relevant documents
retriever = vectorstore.as_retriever()

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answering prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the final retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 5. Start the Chatbot ---
print("\n--- Chatbot is ready! ---")
print("Ask a question or type 'exit' to quit.")

chat_history = []

while True:
    query = input("\nYou: ")
    if query.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    # Get the response from the chain
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    # Update chat history
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": result["answer"]})

    print(f"\nBot: {result['answer']}")


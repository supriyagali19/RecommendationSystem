import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

# --- 1. INITIAL CONFIGURATION & MODEL LOADING ---

# Set up the Streamlit page
st.set_page_config(page_title="AI Recommendation System", page_icon="📚", layout="wide")

# Load environment variables from a .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Use a function to cache models and data loading for better performance
@st.cache_resource
def load_models_and_data():

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load data from CSV
    file_path = "./Data.csv"
    loader = CSVLoader(file_path=file_path)
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    # Create FAISS vector store from documents
    vectordb = FAISS.from_documents(documents, huggingface_embeddings)

    # Initialize the LLM with Groq
    llm = ChatGroq(model="llama3-8b-8192", temperature=0) # Using Llama3 8b for speed and quality

    return vectordb, llm

# --- 2. SETUP RETRIEVAL CHAIN ---

try:
    vectordb, llm = load_models_and_data()

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert recommendation assistant.
        The user is asking for suggestions about resources, tools, articles, or courses on a specific topic.
        Use the provided context to give an accurate and relevant answer. If the context does not contain
        relevant information, politely state that you couldn't find specific resources in your database,
        but you can provide a general answer based on your own knowledge.

        <Context>
        {context}
        </Context>

        User Query: {input}

        Helpful Answer:
        """
    )

    # Create the chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectordb.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

except Exception as e:
    st.error(f"Error initializing the application: {e}")
    st.stop()


# --- 3. STREAMLIT UI LAYOUT ---

# Sidebar content
with st.sidebar:
    st.title("📚 AI Recommendation System")
    st.markdown("""
    This app provides recommendations for learning resources, tools, and articles on various topics.
    
    **How to use:**
    1.  Type your question in the chat input below.
    2.  The assistant will search its database for relevant resources.
    3.  You can view the specific sources used for the answer in the expander.
    
    """)

# Main page title
st.title("Chat with the Recommendation Bot")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! What topic are you interested in learning about today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLE USER INPUT AND GENERATE RESPONSE ---

if prompt := st.chat_input("e.g., 'Suggest some courses for learning about AI'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a spinner while generating the response
    with st.spinner("Searching for recommendations..."):
        try:
            # Invoke the retrieval chain
            response = retrieval_chain.invoke({'input': prompt})
            answer = response.get('answer', "Sorry, I couldn't generate a response.")
            context_docs = response.get('context', [])

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                # If context was used, show it in an expander
                if context_docs:
                    with st.expander("See the sources used for this answer"):
                        for i, doc in enumerate(context_docs):
                            st.info(f"Source {i+1}:\n\n{doc.page_content}")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

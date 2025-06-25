# RecommendationSystem(AI Recommendation System for Learning Resources)
An intelligent conversational AI system built with LangChain, Groq (Llama3), HuggingFace embeddings, and FAISS, providing relevant learning resource recommendations via a Streamlit chat interface, grounded in a RAG architecture


Overview
This project aims to simplify the process of discovering learning resources by offering an interactive chat experience. Users can ask questions about various topics, and the system will provide accurate and context-aware recommendations for courses, articles, tools, and more, leveraging a pre-indexed knowledge base.

Key Features
Natural Language Interaction: Engage with the system using conversational queries.

Semantic Search: Efficiently retrieves semantically similar content using HuggingFace BGE Embeddings and a FAISS vector store.

Intelligent Recommendations: Leverages the llama3-8b-8192 Large Language Model (LLM) from Groq to generate helpful and relevant answers.

Retrieval-Augmented Generation (RAG): Ensures responses are factual and grounded in the provided data, minimizing hallucinations.

Source Transparency: Users can view the exact document chunks that informed the LLM's response.

User-Friendly Interface: Built with Streamlit for an intuitive and responsive web application.

Technologies Used
Python

Streamlit: For building the interactive web UI.

LangChain: Framework for orchestrating LLM applications (document loading, text splitting, retrieval, chains).

Groq: High-speed inference engine for LLMs.

HuggingFace sentence-transformers (BAAI/bge-small-en-v1.5): For generating high-quality text embeddings.

FAISS (Facebook AI Similarity Search): For efficient in-memory vector storage and similarity search.

python-dotenv: For managing environment variables.

Setup and Installation
Follow these steps to get the project up and running on your local machine.

1. Clone the Repository
git clone [https://github.com/supriyagali19/RecommendationSystem.git](https://github.com/supriyagali19/RecommendationSystem.git)
cd RecommendationSystem

2. Create a Virtual Environment (Recommended)
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

(If you don't have a requirements.txt yet, create one with the following content:)

streamlit
langchain-groq
langchain-community
langchain-text-splitters
langchain-core
faiss-cpu
sentence-transformers
python-dotenv

4. Prepare Your API Key
You will need a Groq API key to use their LLM service.

Go to Groq Console and generate an API key.

Create a file named .env in the root directory of your project (the same directory as app.py).

Add your API key to the .env file:

GROQ_API_KEY="your_groq_api_key_here"

Replace "your_groq_api_key_here" with your actual Groq API key.

5. Prepare Your Data
The system uses a CSV file named Data.csv for its knowledge base.

Create a Data.csv file in the root directory of your project.

Populate it with your learning resources. Ensure it has relevant columns that CSVLoader can process (e.g., a column for content, and optional columns for metadata). For example:

title,category,description,url
"Python for Data Science",Programming,"An introductory course covering Python basics for data analysis and machine learning.","[https://example.com/python-course](https://example.com/python-course)"
"Machine Learning Fundamentals",AI,"Learn the core concepts of supervised and unsupervised learning.","[https://example.com/ml-course](https://example.com/ml-course)"
"SQL for Data Analysts",Database,"A comprehensive guide to SQL for data manipulation and querying.","[https://example.com/sql-course](https://example.com/sql-course)"

How to Use
Run the Streamlit Application:

streamlit run app.py

Interact with the Chatbot:

Once the application opens in your web browser, you will see a chat interface.

Type your questions or requests in the input box at the bottom (e.g., "Suggest some courses for learning about AI," "What are some good tools for data visualization?").

The chatbot will process your query and provide recommendations.

You can click on "See the sources used for this answer" to view the specific data chunks from which the LLM drew its information.

Project Structure
├── .env                  # Environment variables (e.g., GROQ_API_KEY)
├── Data.csv # Your dataset of learning resources
├── app.py              # Main Streamlit application and RAG logic
└── README.md             # This file
├── requirements.txt      # Python dependencies

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
If you have any questions or feedback, please open an issue in this repository.

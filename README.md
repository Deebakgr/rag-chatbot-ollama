# RAG Chatbot with Ollama

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Ollama, featuring both Streamlit and Flask interfaces.

## Features

- 🤖 RAG-based question answering
- 🎨 Modern Flask web interface
- 📱 Streamlit chat interface
- 🔍 FAISS vector search
- 🦙 Ollama integration

## Setup Instructions

### 1. Install Ollama and Models
```bash
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Vector Database
```bash
python setup_vectorstore.py
```

### 4. Run the Application

**Flask Web Interface (Recommended):**
```bash
python app_flask.py
```
Then open http://127.0.0.1:5000

**Streamlit Interface:**
```bash
streamlit run app.py
```

## Project Structure

```
Wikipediarag/
├── app.py              # Streamlit interface
├── app_flask.py        # Flask web interface
├── setup_vectorstore.py # Vector database setup
├── dataset.txt         # Your dataset
├── templates/
│   └── index.html      # Flask HTML template
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How it Works

1. **Data Processing**: Loads `dataset.txt` and splits into chunks
2. **Embeddings**: Creates embeddings using `nomic-embed-text` model
3. **Vector Store**: Stores embeddings in FAISS for fast similarity search
4. **Question Answering**: Uses `gemma3:latest` model to generate answers based on retrieved context

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Ollama**: Local LLM inference
- **FAISS**: Vector similarity search
- **Flask**: Web framework
- **Streamlit**: Rapid prototyping interface

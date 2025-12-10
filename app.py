import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot with Ollama Gemma")

@st.cache_resource
def load_rag_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    llm = Ollama(model="gemma3:latest", temperature=0)
    return vectorstore, llm

try:
    vectorstore, llm = load_rag_system()
    st.success("✅ System ready!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Run 'python setup_vectorstore.py' first to create the vector database.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the dataset"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = vectorstore.similarity_search(prompt, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                full_prompt = f"""Use the following context to answer the question.
If you don't know the answer, say you don't know If anyone say Thank You, say Your welcome.

Context: {context}

Question: {prompt}

Answer:"""
                
                answer = llm.invoke(full_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

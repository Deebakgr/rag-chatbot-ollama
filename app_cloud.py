import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot")

# Get OpenAI API key from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to continue.")
        st.stop()

@st.cache_resource
def load_rag_system():
    # Load and process dataset
    loader = TextLoader('./dataset.txt', encoding='utf-8')
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create LLM
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
    
    return vectorstore, llm

try:
    vectorstore, llm = load_rag_system()
    st.success("✅ System ready!")
except Exception as e:
    st.error(f"❌ Error: {e}")
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
If you don't know the answer, say you don't know.

Context: {context}

Question: {prompt}

Answer:"""
                
                response = llm.invoke(full_prompt)
                answer = response.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

app = Flask(__name__)

vectorstore = None
llm = None

def load_rag_system():
    global vectorstore, llm
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    llm = Ollama(model="gemma3:latest", temperature=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""
        
        answer = llm.invoke(prompt)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_rag_system()
    app.run(debug=True)

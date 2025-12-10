from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading dataset...")
loader = TextLoader('./dataset.txt', encoding='utf-8')
docs = loader.load()

print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"Creating embeddings for {len(splits)} chunks...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Building vector store...")
vectorstore = FAISS.from_documents(splits, embeddings)

print("Saving vector store...")
vectorstore.save_local("vectorstore")

print("✅ Setup complete!")

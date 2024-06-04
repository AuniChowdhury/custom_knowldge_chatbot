from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

import pickle


os.environ["OPENAI_API_KEY"] = "API_KEY"

loader = UnstructuredPDFLoader("sample.pdf")
raw_documents = loader.load()
print(raw_documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)



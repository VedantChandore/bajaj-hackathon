from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import os

# Environment
os.environ["COHERE_API_KEY"] = "Kwi33HNnmXRDCkO4j7FndNP3LATOoKX3yvoOdztK"

# Load PDF
pdf_path = "/Users/vedant/Downloads/HDFHLIP23024V072223.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

import cohere
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
embeddings = CohereEmbeddings(client=cohere_client, model="embed-english-v3.0") # type: ignore


# Vector Store
vectorstore = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Cohere LLM
llm = Cohere(model="command-r-plus", temperature=0.2, input_type="prompt")

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Questions
questions = [
    "Does this policy cover day care procedures or surgeries?",
    "What is the waiting period for pre-existing diseases (PED)?",
    "Are expenses for the organ donor covered during transplant surgeries?",
    "What is the grace period for premium payment on renewal?",
    "Does the policy provide maternity coverage for normal and C-section deliveries?",
    "What is the Stay Active benefit and how does it affect the renewal premium?",
    "What ambulance expenses are covered under this policy?",
    "Are cancer and heart attack covered under the critical illness benefits?",
    "Does this policy cover AYUSH treatments like Ayurveda and Homeopathy?",
    "Is there a No Claim Bonus or cumulative bonus provided?"
]

# Run the RAG system
for q in questions:
    print(f"\n🔎 Question: {q}")
    result = qa_chain.invoke(q)
    print("✅ Answer:", result['result'])

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,PyPDFium2Loader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import UnstructuredPDFLoader

# Set your Groq API key
GROQ_API_KEY = 'ENTER YOUR API KEY '

print("=" * 70)
print("RAG APPLICATION - PDF Q&A with Groq LLM")
print("=" * 70)

# STEP 1: Load PDF
print("\n[1/5] Loading PDF...")
pdf_path = input("enter your pdf path:")  # Replace with your PDF path
# loader = PyPDFium2Loader(pdf_path, mode="page") # Original line

if os.path.exists(pdf_path):
    print(" File found!")
else:
    print(" File not found:", pdf_path)

loader = PyPDFLoader(pdf_path) # Changed to PyPDFLoader
#loader = UnstructuredPDFLoader(pdf_path, mode="elements")
documents = loader.load()
print(f" Loaded {len(documents)} pages from PDF")

# STEP 2: Split text into chunks using RecursiveCharacterTextSplitter
print("\n[2/5] Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each chunk
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
)
chunks = text_splitter.split_documents(documents)
print(f" Created {len(chunks)} text chunks")

# STEP 3: Create embeddings using HuggingFace
print("\n[3/5] Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print(" Embeddings model loaded")

# STEP 4: Create vector store
print("\n[4/5] Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)   #  storing the embedding vector in the database
print(" Vector store created")

# STEP 5: Initialize Groq LLM
print("\n[5/5] Setting up Groq LLM...")
llm = ChatGroq(
    model="openai/gpt-oss-120b", #openai/gpt-oss-120b
    temperature=0.3,
    max_tokens=2048,
    api_key=GROQ_API_KEY
)
print(" LLM initialized")

# Create custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
k=int(input("enter your top result count: "))
# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": k}),  # Retrieve top 3 chunks
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("\n" + "=" * 20)
print("Ask questions about your PDF!")
print("=" * 70)

# Example queries
queries = [
    "What is the main topic of this document?",
    "Can you summarize the key points?",
    # Add your own questions here
]

for i, query in enumerate(queries, 1):
    print(f"\n[Question {i}]: {query}")
    print("-" * 20)

    result = qa_chain.invoke({"query": query})

    print(f"Answer: {result['result']}")
    print(f"\nSources (Top {len(result['source_documents'])} relevant chunks):")
    for j, doc in enumerate(result['source_documents'], 1):
        print(f"  [{j}] Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}...")

# Interactive Q&A loop
print("\n" + "=" * 50)
print(" Type 'quit' to exit")
print("=" * 50)

while True:
    question = input("\nYour question: ").strip()

    if question.lower() in ['quit', 'exit', 'q','out']:
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nThinking...")
    result = qa_chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")
    print(f"\nRelevant sources: {len(result['source_documents'])} chunks used")

# Optional: Save vector store for later use
print("\n" + "=" * 70)
print("Saving vector store...")
vectorstore.save_local("faiss_index")
print(" Vector store saved to 'faiss_index' directory")
print("\nTo load it later, use:")
print("vectorstore = FAISS.load_local('faiss_index', embeddings)")
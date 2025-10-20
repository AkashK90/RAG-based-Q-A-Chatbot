import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Set your Groq API key
GROQ_API_KEY = 'ENTER YOUR API KEY'

# Streamlit interface
st.title("PDF Chatbot with Groq LLM")

# Sidebar for configurations
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    k = st.number_input("Number of relevant chunks (top results)", min_value=1, max_value=10, value=3)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Process uploaded file
if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""] # for priority
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Create vector store
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

        # Initialize Groq LLM
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=2048,
            api_key=GROQ_API_KEY
        )

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

        # Create RetrievalQA chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    st.sidebar.success("Document processed successfully!")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.qa_chain is not None:
    if prompt := st.chat_input("Ask a question about the loaded document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                response = result['result']
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a document to start chatting.")
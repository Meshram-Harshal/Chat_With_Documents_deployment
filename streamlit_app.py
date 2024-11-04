import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
import os
from pandasai.llm.local_llm import LocalLLM
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from bs4 import BeautifulSoup
import requests

# Retrieve GROQ API Key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=0.2
)

# Function to process PDF files
def process_pdfs(files):
    texts = []
    metadatas = []
    for file in files:
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

# Function to process website content
def process_website(url):
    # Scrape website text using BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the website (you can modify this to be more specific)
    website_text = soup.get_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(website_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"Website {i}"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

# Function to chat with CSV data
def chat_with_csv(df, query):
    # Initialize LocalLLM with Meta Llama 3 model
    llm = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3")
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Multi-file & Website ChatApp powered by LLM")

# Select input type for upload or website input
input_type = st.sidebar.radio("Select input type", ('PDF', 'CSV', 'Website URL'))

if input_type == 'PDF':
    input_files = st.sidebar.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if input_files:
        # Process PDFs
        @st.cache_resource
        def cached_process_pdfs(files):
            return process_pdfs(files)
        
        chain = cached_process_pdfs(input_files)
        st.success(f"Processing {len(input_files)} PDF files done. You can now ask questions!")
        st.session_state.chain = chain

        if 'chain' in st.session_state:
            user_query = st.text_input("Ask a question about the PDFs:")
            if user_query:
                chain = st.session_state.chain
                res = chain.invoke(user_query)
                answer = res["answer"]

                # Display only the answer
                st.write(answer)

elif input_type == 'CSV':
    input_files = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

    if input_files:
        selected_file = st.selectbox("Select a CSV file", [file.name for file in input_files])
        selected_index = [file.name for file in input_files].index(selected_file)

        # Load and display the selected CSV file
        st.info("CSV uploaded successfully")
        data = pd.read_csv(input_files[selected_index])
        st.dataframe(data.head(3), use_container_width=True)

        # Enter the query for analysis
        st.info("Chat Below")
        input_text = st.text_area("Enter the query")

        # Perform analysis
        if input_text:
            if st.button("Chat with CSV"):
                st.info("Your Query: " + input_text)
                result = chat_with_csv(data, input_text)
                st.success(result)

elif input_type == 'Website URL':
    # Create a container for the website URL input
    with st.container():
        st.write("Enter the website URL:")
        url = st.text_input("Website URL")
        
        if url:
            # Process website
            @st.cache_resource
            def cached_process_website(url):
                return process_website(url)
            
            chain = cached_process_website(url)
            st.success("Website content processed successfully. You can now ask questions!")
            st.session_state.chain = chain

            if 'chain' in st.session_state:
                user_query = st.text_input("Ask a question about the website:")
                if user_query:
                    chain = st.session_state.chain
                    res = chain.invoke(user_query)
                    answer = res["answer"]

                    # Display only the answer
                    st.write(answer)

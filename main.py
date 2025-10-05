import os
import streamlit as st
import pickle
import time
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.news import NewsURLLoader


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Readify:Your Personal Website Reader")
st.sidebar.title("Paste links here")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",  
    temperature=0.9,
    max_tokens=500,
    api_key="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",
    # api_key=os.getenv("DEEPSEEK_API_KEY"),      
    base_url="https://openrouter.ai/api/v1"   
)


if process_url_clicked:

    loader = NewsURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    #embeddings = OpenAIEmbeddings()
    if not docs:
        st.error("❌ No content could be extracted from the provided URLs. Please check if URLs are valid and accessible.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"  
            )
            
            result = chain({"query": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["result"])
            

        
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)





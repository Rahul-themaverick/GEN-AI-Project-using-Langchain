## Generative AI (RAG-based) Web Reader using LangChain and Deepseek API

# Readify: Web Research Tool 

Readify is a user-friendly web research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the any articles.

# Features

- Load URLs text files to fetch article content.
- Process article content through LangChain's NewsURLLoader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to     enable swift and effective retrieval of relevant information
- Interact with the LLM's (Deepseek) by inputting queries and receiving answers.

- ## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.

  ## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles
- I have used the following news articles

  -https://www.moneycontrol.com/news/business/markets/these-smallcaps-gain-between-10-33-as-broader-indices-outperform-13596800.html

  -https://www.moneycontrol.com/news/business/markets/nse-reduces-f-o-lot-sizes-for-nifty-50-and-three-other-indices-from-october-28-13597333.html

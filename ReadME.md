# Constitution of Ukraine RAG System

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering user questions using the 
**Constitution of Ukraine** as a knowledge base.


## 💡 Project Overview
The system takes a user question and responds using relevant excerpts from the Constitution of Ukraine. It consists of:

- Chunking the Constitution text
- Embedding and indexing chunks
- Retrieving relevant chunks based on cosine similarity
- Re-ranking results using a cross-encoder
- Generating final answers using an LLM
- Displaying the answer and metadata (scores and chunks)

## 🏗️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/RailwaymanUZ/Constitution_of_Ukraine_RAG_System
   cd Constitution_of_Ukraine_RAG_System
   ```
2. Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```
3. In the root directory of the project, create a .env file with OPENAI_API_KEY:

    ```bash
   OPENAI_API_KEY=your_openai_api_key_here
    ```   
    Or set the environment variable in your system: 
    ```bash
    export OPENAI_API_KEY=your_openai_api_key_here
    ```
4. Run the CLI application:
    ```bash
   python main.py
   ```
5. When prompted:<br>
\*\*Будь ласка введіть своє питання\*\*:<br>
send your question<br>


## 📄 How Chunking Was Done
The Constitution was **chunked by article** instead of by token count, sentence, or character limit. 
This approach was chosen because:

- Each article is semantically complete.
- Legal documents are structured naturally by articles.
- It improves retrieval quality without splitting coherent thoughts.

## 🧠 Embeddings and Indexing
- Embedding model: OpenAI embeddings
- Indexing method: FAISS (FlatL2 Index)
- Retrieval method: Maximal Marginal Relevance (MMR)

## 🔍 Re-Ranking
We used a cross-encoder to re-rank the top chunks based on their relevance to the query.

- Weak point: The current cross-encoder occasionally produces negative scores, indicating irrelevant.
- This is a limitation of the model and can be improved in future versions.

## 📊 Table test to path
notebooks/test_result.csv

## 📈 Evaluation & Observations
#### Cross-encoder weakness:
- Some answers had negative scores.
- Indicates that less relevant documents were mistakenly prioritized.

#### Cosine similarity worked well:
- No negative scores.
- Reliable initial filter for relevant content.

#### Recommendations:
- Try alternative cross-encoders for re-ranking.
- Consider eliminating cross-encoder entirely.
- Build a custom LLM-based re-ranker that scores (0–100) how well a chunk answers a query.



### 🧑‍💻 Developer
Valera Lemeshko<br>
Email - lemeshkovalerij@gmail.com


# AI Shopping Assistant

This is a local Python project that implements a basic AI shopping assistant using Streamlit, LangChain, and Groq LLMs. It features product searches, customer ratings, simulated order placement, user preferences memory, and policy lookups using local Vector RAG.

---

## 📦 What is in the Project

The codebase consists of:

### 1. Database & Catalog
*   **SQLite Database (`store.db`)**: Stores catalog items, customer reviews, order logs, and user preferences.
*   **Products Catalog**: Features 32 items across categories like honey, cooking oils, nuts, grains, tea, coffee, snacks, and milk alternatives.
*   **Customer Reviews**: Contains simulated reviews used to calculate average star ratings for products.

### 2. Search & Retrieval (RAG)
*   **Unstructured Knowledge (`store_info.txt`)**: A text file containing store policies (returns, shipping) and product care/storage instructions.
*   **Vector Search Index (`faiss_index/`)**: A local vector database built using FAISS and the `all-MiniLM-L6-v2` sentence-transformer model to search the policies text.

### 3. AI Agent & Guardrails
*   **Text Agent (`shopping_agent.py`)**: Uses Groq's Qwen 32B model to reason about user input and call appropriate database/vector tools.
*   **Vision search**: Uses Llama 17B vision model to describe uploaded product images and find similar items in the catalog.
*   **Input Guardrail**: A simple Python wrapper that filters out off-topic messages (like general knowledge or math queries) while fast-tracking greetings and order selections.

### 4. Frontend Interface
*   **Streamlit Web App (`app.py`)**: A chat web interface styled with custom CSS (glassmorphism chat bubbles, Outfit font, and slate color styling).

### 5. Automated Evaluation
*   **Evals Runner (`run_evals.py`)**: A script that runs test queries through the agent to check if the correct tools are invoked and responses meet format guidelines.

---

## 🛠️ Requirements & Libraries
Make sure you have Python 3.10+ installed. The dependencies used are:
*   `streamlit`
*   `python-dotenv`
*   `langchain` / `langchain-core` / `langchain-community`
*   `langchain-groq` (to interface with Qwen & Llama models)
*   `faiss-cpu` (for local vector retrieval)
*   `sentence-transformers` (to create embeddings locally)

---

## ⚙️ Setup & How to Run

### Step 1: Clone and install dependencies
```bash
git clone <your-repo-link>
cd Shopping_Agent_Project
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a file named `.env` in the root folder and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 3: Populate the Database
Run the SQLite database initialization script:
```bash
python setup_db.py
```

### Step 4: Build the RAG Index
Compile the local FAISS vector database from the policies text:
```bash
python setup_vector_db.py
```

### Step 5: Launch the Streamlit App
Start the local web server:
```bash
streamlit run app.py
```

### Step 6: (Optional) Run Evaluations
To verify tool-calling routing and formatting:
```bash
python run_evals.py
```

---

## 📁 Key File Descriptions
*   `app.py`: Streamlit frontend layout, guardrails, and message loops.
*   `shopping_agent.py`: LangChain tools, LLM configs, database helpers, and prompt rules.
*   `setup_vector_db.py`: Scripts to parse `store_info.txt`, embed text, and save the local FAISS index.
*   `store_info.txt`: Text guidelines for returns, shipping, and product care.
*   `reviews_api.py`: Python module to read ratings from the SQLite DB.

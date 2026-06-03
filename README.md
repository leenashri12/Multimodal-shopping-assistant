# 🛒 AI Shopping Assistant

An intelligent, AI-powered e-commerce shopping assistant web application built with **Streamlit**, **LangChain**, and **Groq**. 

This application allows users to search for products in a local database using text queries or by uploading images (multimodal vision search). The assistant can also check product ratings and place orders (writes directly to the SQLite database).

---

## 🚀 Features

- **Text Chat Interface**: Ask for products using natural language (e.g., *"I want organic honey under $15 with a 4+ rating"*).
- **Shop by Image (Vision)**: Upload an image of a product, and the assistant will analyze it using a multimodal vision model to search the store for matching items.
- **Rating Lookup**: Automatically retrieves average reviews and ratings for matched products.
- **Simulated Checkout**: Place orders directly through the chat. The assistant will ask for confirmation before modifying the database.
- **Persistent Personalization**: Automatically remembers user preferences (such as organic preferences or price limits) across sessions.
- **Order History Memory**: Lets you query and view all your previously placed orders by asking *"what have I ordered before?"*.
- **Input Guardrails**: Intercepts and politely redirects off-topic queries (e.g., poetry, general knowledge) to protect the LLM and prevent token wastage.

---

## 🛠️ Tech Stack

- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **AI Agent Framework**: [LangChain](https://www.langchain.com/)
- **LLM Provider**: [Groq](https://groq.com/)
  - **Text Agent Model**: `qwen/qwen3-32b`
  - **Vision/Image Search Model**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Database**: SQLite (local `store.db` file)

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Shopping_Agent_Project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Initialize the Database
Run the seed script to create the tables and populate them with products and reviews:
```bash
python setup_db.py
```

### 6. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📁 File Structure

- `app.py`: The Streamlit web UI with safety guardrails and rate-limit retry protection.
- `shopping_agent.py`: Agent logic, LangChain tool definitions, preference database logic, and LLM configuration.
- `reviews_api.py`: Python interface to query product reviews from the SQLite database.
- `setup_db.py`: Database initialization script containing sample products, reviews, and preference table creations.
- `run_evals.py`: Automated evaluation script for testing tool call accuracy and response quality using an LLM-as-judge with rate-limit recovery.
- `resources/`: Folder containing example images for image search.
- `.gitignore`: Specifies files that Git should ignore (e.g., virtual environments, database files, and `.env` files containing API keys).
- `requirements.txt`: Python package dependencies.

---

## 🧪 Running Evaluations

To run the automated evaluation test suite verifying tool call accuracy and response quality:
```bash
python run_evals.py
```


import base64
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_retries=6)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

# LangChain FAISS and HuggingFaceEmbeddings are lazy-imported to speed up Streamlit re-execution

# Paths for vector index
DB_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

# Lazy loading helpers to prevent loading PyTorch weights on every import
embeddings_model = None
vector_store = None

def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

def get_vector_store():
    global vector_store
    if vector_store is None:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                from langchain_community.vectorstores import FAISS
                model = get_embeddings_model()
                vector_store = FAISS.load_local(FAISS_INDEX_PATH, model, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Warning: Failed to load FAISS index: {e}")
    return vector_store


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic, average_rating, review_count.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = """
        SELECT p.id, p.name, p.category, p.price, p.description, p.is_organic,
               AVG(r.rating) as average_rating,
               COUNT(r.id) as review_count
        FROM products p
        LEFT JOIN reviews r ON p.id = r.product_id
        WHERE 1=1
    """
    params: list = []

    if query:
        sql += " AND (p.name LIKE ? OR p.description LIKE ? OR p.category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND p.price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND p.is_organic = ?"
        params.append(1 if is_organic else 0)

    sql += " GROUP BY p.id"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":             row[0],
            "name":           row[1],
            "category":       row[2],
            "price":          row[3],
            "description":    row[4],
            "is_organic":     bool(row[5]),
            "average_rating": round(row[6], 2) if row[6] is not None else 0.0,
            "review_count":   row[7] if row[7] is not None else 0,
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def get_order_history() -> str:
    """
    Retrieve the history of all orders placed by the user.
    Returns a JSON array of orders, each containing: id, product_id, product_name, price, ordered_at.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, product_id, product_name, price, ordered_at FROM orders ORDER BY ordered_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    orders = [
        {
            "id": row[0],
            "product_id": row[1],
            "product_name": row[2],
            "price": row[3],
            "ordered_at": row[4]
        }
        for row in rows
    ]
    return json.dumps(orders)


@tool
def get_user_preferences_tool() -> str:
    """
    Retrieve all saved user preferences (such as organic preference or price limits).
    Returns a JSON object containing keys like 'prefers_organic', 'max_price', etc.
    Always call this at the beginning of the conversation if you haven't already.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS user_preferences (pref_key TEXT PRIMARY KEY, pref_value TEXT)")
    cursor.execute("SELECT pref_key, pref_value FROM user_preferences")
    rows = cursor.fetchall()
    conn.close()
    
    prefs = {row[0]: row[1] for row in rows}
    return json.dumps(prefs)


@tool
def save_user_preference(pref_key: str, pref_value: str) -> str:
    """
    Save or update a user preference (e.g. key='prefers_organic' value='True', key='max_price' value='20').
    Use this when the user explicitly expresses a general preference (e.g., 'I only buy organic', 'My budget is always under $20').
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS user_preferences (pref_key TEXT PRIMARY KEY, pref_value TEXT)")
    cursor.execute("INSERT OR REPLACE INTO user_preferences (pref_key, pref_value) VALUES (?, ?)", (pref_key, pref_value))
    conn.commit()
    conn.close()
    return f"Preference '{pref_key}' saved as '{pref_value}'."


@tool
def search_policy_and_faq(query: str) -> str:
    """
    Search the store's unstructured guidelines, return policy, shipping rates, and product care guidelines.
    Use this tool when the user asks questions about return policies, refund durations, shipping costs,
    support email, business hours, shipping destinations, or how to store/care for products like honey, oil, tea, coffee, nuts, seeds, and milk.
    """
    v_store = get_vector_store()
    if not v_store:
        return (
            "Error: Vector database FAQ and policy index is not loaded. "
            "Please run 'python setup_vector_db.py' to generate the index first."
        )

    # Perform semantic similarity search (retrieve top 2 matching chunks)
    docs = v_store.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


def is_shopping_related(user_message: str) -> bool:
    """
    Classify whether a user message is shopping-related or not.
    Returns True if it is, False if it is off-topic.
    """
    if user_message.startswith("I uploaded a product image"):
        return True

    # Simple check for common greeting words/phrases to prevent blocking them
    cleaned = user_message.strip().lower().rstrip("?.!")
    greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup"}
    conversational = {"how are you", "who are you", "what can you do", "what are you", "help", "menu", "info"}
    
    if (
        cleaned in greetings 
        or cleaned in conversational
        or any(cleaned.startswith(g + " ") for g in greetings)
        or any(cleaned.startswith(c + " ") for c in conversational)
    ):
        return True

    prompt = (
        "You are an input guardrail for a shopping assistant.\n"
        "Your task is to classify whether the user's message is related to shopping, products, orders, "
        "reviews, store items, or shopping preferences.\n\n"
        f"User Message: \"{user_message}\"\n\n"
        "Respond with ONLY 'yes' or 'no'. Do not include any punctuation, explanation, or extra words."
    )
    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        return "yes" in decision
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def get_user_preferences_from_db() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS user_preferences (pref_key TEXT PRIMARY KEY, pref_value TEXT)")
        cursor.execute("SELECT pref_key, pref_value FROM user_preferences")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


_agent_compiled = create_agent(
    tools=[
        search_products,
        get_rating,
        checkout,
        describe_product_image,
        get_order_history,
        get_user_preferences_tool,
        save_user_preference,
        search_policy_and_faq
    ],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "USER PREFERENCES:\n"
        "1. Check the system instructions or conversation context (SystemMessage) for saved user preferences. If they are already provided, apply them (e.g. prefers_organic='True' means only search for organic products unless overridden). Otherwise, call get_user_preferences_tool to fetch them.\n"
        "2. Apply these saved preferences (e.g., preferring organic products or maximum price limit) to search_products automatically unless the user explicitly overrides them.\n"
        "3. When the user explicitly mentions a preference (e.g., 'I always buy organic', 'my limit is $20', 'never show me things over $15'), call save_user_preference to save it for future sessions.\n\n"
        "ORDER HISTORY:\n"
        "1. When the user asks about their previous orders, what they have bought before, or order history, call get_order_history to fetch and list their orders.\n\n"
        "STORE POLICIES & CARE GUIDELINES (RAG):\n"
        "1. When the user asks about shipping times, shipping costs, support emails, customer support hours, return policies, refund timelines, or product care/storage guidelines (e.g. how to store honey, cooking oil, coffee, tea, nuts, or milk), call search_policy_and_faq to get the accurate information.\n"
        "2. Answer the user's policy or care question using ONLY the facts retrieved by the tool. Be direct and friendly.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given, as well as saved preferences).\n"
        "2. Note: search_products already returns average_rating and review_count. You do NOT need to call get_rating separately for each search result candidate. Only call get_rating if you need to double-check ratings or if specifically asked for rating details.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Never place an order directly from the user's initial message. Even if the user says 'I want to order X' in their first query, you must first search the catalog, display the item in the standard numbered list, and ask for verification. Only call checkout when the user confirms with 'yes' or confirms a listed item number.\n"
        "2. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "3. Call checkout with that product_id (the number from (ID:X)).\n"
        "4. Confirm the order to the user in plain text.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message."
    ),
)


class PreferencesWrapper:
    def invoke(self, input_dict: dict, config=None):
        prefs = get_user_preferences_from_db()
        messages = input_dict.get("messages", [])
        pref_context = f"Saved User Preferences: {json.dumps(prefs)}"
        new_messages = [{"role": "system", "content": pref_context}] + messages
        return _agent_compiled.invoke({"messages": new_messages}, config)


agent = PreferencesWrapper()

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)

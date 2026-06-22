from flask import Flask, render_template, request
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
import os

app = Flask(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------

embeddings = download_hugging_face_embeddings()

DB_FAISS_PATH = "vectorstore/db_faiss"

# ---------------------------------------------------
# Load FAISS Index
# ---------------------------------------------------

if os.path.exists(DB_FAISS_PATH):

    docsearch = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Loaded existing FAISS index.")

else:

    print("No local index found. Creating index...")

    extracted_data = load_pdf_file("data/")
    text_chunks = text_split(extracted_data)

    docsearch = FAISS.from_documents(
        text_chunks,
        embeddings
    )

    docsearch.save_local(DB_FAISS_PATH)

    print(f"FAISS index created and saved to {DB_FAISS_PATH}")

# ---------------------------------------------------
# Retriever
# ---------------------------------------------------

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------------------------------------------
# Groq LLM
# ---------------------------------------------------

chatModel = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    return_messages=False
)

# ---------------------------------------------------
# Chat Chain (Greetings, Thanks, Bye etc.)
# ---------------------------------------------------

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Ada, a friendly AI health assistant.

Rules:
- Your name is Ada.
- Introduce yourself ONLY when the user greets you for the first time.
- If the user already knows your name or directly calls you Ada, do not introduce yourself again.
- Never repeatedly say "I am Ada".
- Be warm, friendly and professional.
- Keep responses concise.
- If the user says "No", "No Thanks", "Nothing", "That's all", or similar, politely end the conversation.
- Do not ask another question after the user indicates they are finished.

Examples:

User: Hi
Assistant: Hello! 👋 I'm Ada, your AI health assistant. How can I help you today?

User: Hello
Assistant: Hello! 👋 I'm Ada, your AI health assistant. How can I help you today?

User: Hi Ada
Assistant: Hello! 👋 How can I help you today?

User: Ada
Assistant: Hi! 👋 How can I assist you today?

User: Thanks
Assistant: You're very welcome! 😊 Let me know if there's anything else I can help you with.

User: Bye
Assistant: Goodbye! 👋 Take care and stay healthy.

User: How are you?
Assistant: I'm doing great, thank you for asking! 😊 How can I help you today?

User: Ok Thanks
Assistant: Glad I could help! 😊 Wishing you a great day ahead.

User: No Thanks
Assistant: You're welcome! 😊 If you need any help in the future, feel free to ask.

User: No thank you
Assistant: No problem! 😊 I'm here whenever you need assistance.

User: No
Assistant: Alright! 😊 Feel free to reach out if you have any questions later.

User: Nothing
Assistant: That's perfectly fine! 😊 Have a great day and take care.

User: That's all
Assistant: Glad I could help! 😊 Take care and stay healthy.

User: Bye
Assistant: Goodbye! 👋 Take care and stay healthy.

User: See you
Assistant: See you later! 👋 Wishing you a wonderful day.
"""
        ),
        ("human", "{input}")
    ]
)

chat_chain = chat_prompt | chatModel

# ---------------------------------------------------
# Medical RAG Prompt
# ---------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    print(f"User Input: {msg}")

    msg_lower = msg.lower().strip()

    # Greetings / Small Talk
    chat_keywords = [
        "hi",
        "hello",
        "hey",
        "hii",
        "thanks",
        "thank you",
        "ok thanks",
        "okay thanks",
        "thx",
        "bye",
        "goodbye",
        "see you",
        "how are you",
        "what's up",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if any(keyword in msg_lower for keyword in chat_keywords):

        response = chat_chain.invoke({"input": msg})

        print("Chat Response:", response.content)

        return response.content

    # Load memory
    memory_variables = memory.load_memory_variables({})

    chat_history = memory_variables.get("chat_history", "")

    # Create enhanced query using memory
    enhanced_query = f"""
Previous messages:

{chat_history}

Question:
{msg}
"""

    # Run RAG
    response = rag_chain.invoke({"input": enhanced_query})

    # Save conversation to memory
    memory.save_context(
        {"input": msg},
        {"output": response["answer"]}
    )

    print("RAG Response:", response["answer"])

    return response["answer"]


# ---------------------------------------------------
# Run App
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )
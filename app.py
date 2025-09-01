import os
from dotenv import load_dotenv
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_template

#Load API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embeddings & Index
embeddings = download_embeddings()
index_name = "medical-chatbot"

docsearch2 = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch2.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

#LLM & Prompt
llm = ChatOpenAI(model_name="gpt-4o", streaming=True)  # enable streaming
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{input}"),
    ]
)

# Streamlit UI
st.set_page_config(page_title="🩺 Medical Chatbot", page_icon="💬", layout="wide")

st.title("🩺 Medical Chatbot")
st.markdown(
    "This chatbot allows you to ask **medical-related questions**. "
    "It retrieves information from uploaded medical PDFs and provides AI-powered answers."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.slider("Number of retrieved docs", 1, 5, 3)
    retriever.search_kwargs["k"] = k_value
    st.markdown("---")
    st.markdown("💡 Powered by **LangChain + Pinecone + OpenAI**")

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat Display
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if user_input := st.chat_input("Ask me a medical question..."):
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Retrieve context
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."

    # Prepare prompt
    formatted_prompt = prompt.format_messages(input=user_input, context=context)

    # Streaming LLM Response
    with st.chat_message("assistant"):
        response_stream = llm.stream(formatted_prompt)
        bot_reply = st.write_stream(response_stream)

    # Save final response
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})


"""
Local Streamlit tutor app with
1. TiDB login (modal)
2. Gemini streaming chat **with learning-style prompt**
3. Background call to HF Space /predict for topic detection
4. Inline quiz generator
"""
import os
import uuid

import streamlit as st
import requests
import mysql.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ----------------- CONFIG -----------------
load_dotenv()
API_KEY      = os.getenv("GOOGLE_API_KEY")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")
MODEL_NAME   = "gemini-2.0-flash-exp"

if not all([API_KEY, HF_SPACE_URL]):
    st.error("❌ Missing required environment variables (see .env).")
    st.stop()

# ----------------- DB HELPER -----------------
@st.cache_resource
def get_db_conn():
    return  mysql.connector.connect(
            host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
            port=4000,
            user="4V44XYoMA7okY9v.root",
            password="aW2CrSwcTgjFhNAb",
            database="final_project",
           
        )
def student_exists(student_id: str) -> bool:
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM student_performance WHERE student_id = %s LIMIT 1", (student_id,))
        return cur.fetchone() is not None
    except Exception as e:
        st.error(f"DB error: {e}")
        return False

def get_learning_style(student_id: str) -> str:
    """Return 'Quick learner', 'Regular student', or 'Less active' from DB."""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT learning_style FROM student_performance WHERE student_id = %s LIMIT 1", (student_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else "Regular student"
    except Exception:
        return "Regular student"

# ----------------- LOGIN MODAL -----------------
def login_modal():
    @st.dialog("🔐 Student Login", width="small")
    def _modal() -> None:
        sid = st.text_input("Student ID", max_chars=25, key="login_sid")
        if st.button("Login", use_container_width=True):
            if student_exists(sid):
                st.session_state["student_id"] = sid
                st.rerun()
            else:
                st.error("Invalid Student ID.")

    if "student_id" not in st.session_state:
        _modal()
        st.stop()

# ----------------- PROMPT BUILDER -----------------
def build_system_prompt(style: str) -> str:
    if style == "Quick learner":
        return (
            "You are an expert-level academic coach. Provide concise, direct explanations "
            "and challenge the student with advanced follow-ups."
        )
    elif style == "Less active":
        return (
            "You are a patient, encouraging tutor. Break topics into tiny steps, "
            "use simple language, and ask low-stakes questions."
        )
    else:  # Regular student
        return (
            "You are a friendly tutor. Explain step-by-step with analogies "
            "and check understanding with open questions."
        )

# ----------------- GEMINI HELPERS -----------------
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY)

def create_chat_session(style: str):
    cfg = types.GenerateContentConfig(system_instruction=build_system_prompt(style))
    return get_gemini_client().chats.create(model=MODEL_NAME, config=cfg)

def topic_detect(student_id: str, text: str):
    try:
        requests.post(HF_SPACE_URL, json={"student_id": student_id, "text": text}, timeout=3)
    except Exception as e:
        print("Topic detection call failed:", e)

# ----------------- CHAT STATE -----------------
def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.setdefault("chats", {})
    style = get_learning_style(st.session_state["student_id"])
    st.session_state["chats"][cid] = {
        "title": "New Chat",
        "chat": create_chat_session(style),
        "messages": [],
        "style": style
    }
    st.session_state["active_chat"] = cid
    return cid

# ----------------- UI FLOW -----------------
st.set_page_config(page_title="School Tutor", layout="wide")

# 1. Login
login_modal()

# 2. Header
st.title("📚 School Tutor")
st.caption(f"Logged in as **{st.session_state['student_id']}** (style: *{get_learning_style(st.session_state['student_id'])}*)")

# 3. Sidebar
with st.sidebar:
    st.header("Chats")
    if st.button("➕ New Chat"):
        new_chat()

    chat_ids = list(st.session_state.get("chats", {}))
    if not chat_ids:
        new_chat()
        chat_ids = list(st.session_state["chats"])

    labels = [st.session_state["chats"][cid]["title"] for cid in chat_ids]
    idx = chat_ids.index(st.session_state.get("active_chat", chat_ids[0]))
    sel = st.selectbox("Select chat", options=labels, index=idx)
    active_chat_id = chat_ids[labels.index(sel)]

    if st.button("🗑️ Delete selected"):
        st.session_state["chats"].pop(active_chat_id, None)
        st.rerun()

# 4. Chat window
chat = st.session_state["chats"][active_chat_id]

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Dual row: chat input + quiz button
col_input, col_btn = st.columns([5, 1])

with col_input:
    prompt = st.chat_input("Ask me anything...")

with col_btn:
    if st.button("🧩 Quiz", key="quiz_btn"):
        last_topic = chat["messages"][-1]["content"] if chat["messages"] else "general topic"
        quiz_prompt = (
            f"Generate a short 3-question quiz (multiple-choice) about the last discussed topic: {last_topic}. "
            "Make it slightly challenging but memorable."
        )
        try:
            quiz_reply = chat["chat"].send_message(quiz_prompt).text
            chat["messages"].append({"role": "assistant", "content": quiz_reply})
            st.rerun()
        except Exception as e:
            st.error(f"Quiz generation failed: {e}")

if prompt:
    chat["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    topic_detect(st.session_state["student_id"], prompt)

    try:
        stream = chat["chat"].send_message_stream(prompt)
        reply = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in stream:
                txt = getattr(chunk, "text", "")
                if txt:
                    reply += txt
                    placeholder.markdown(reply + "▌")
            placeholder.markdown(reply)
        chat["messages"].append({"role": "assistant", "content": reply})

        if chat["title"] == "New Chat":
            chat["title"] = prompt[:40].replace("\n", " ")
    except Exception as e:
        st.error(f"LLM error: {e}")
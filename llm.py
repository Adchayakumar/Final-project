
"""
Local Streamlit tutor app with
1. TiDB login (modal)
2. Gemini streaming chat
3. Background call to HF Space /predict for topic detection
"""
import os
import time
import uuid
from typing import Dict, Any

import streamlit as st
import requests
import mysql.connector
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ----------------- CONFIG -----------------
load_dotenv()
API_KEY           = os.getenv("GOOGLE_API_KEY")         # optional
HF_SPACE_URL      = os.getenv("HF_SPACE_URL")            # https://xxx.hf.space/run/predict
MODEL_NAME        = "gemini-2.0-flash-exp"

if not all([API_KEY, HF_SPACE_URL]):
    st.error("❌ Missing required environment variables (see .env).")
    st.stop()

# ----------------- DB HELPER -----------------
@st.cache_resource
def get_db_conn():
    return mysql.connector.connect(
            host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
            port=4000,
            user="4V44XYoMA7okY9v.root",
            password="aW2CrSwcTgjFhNAb",
            database="final_project",
            ssl_verify_cert=True,
            ssl_verify_identity=True
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

# ----------------- LOGIN MODAL -----------------
def login_modal():
    """
    Streamlit-native 'modal' via st.dialog.
    Returns student_id iff verified.
    """
    @st.dialog("🔐 Student Login", width="small")
    def _modal() -> None:
        st.write("Enter your Student ID to continue.")
        sid = st.text_input("Student ID", max_chars=25, key="login_sid")
        if st.button("Login", use_container_width=True):
            if student_exists(sid):
                st.session_state["student_id"] = sid
                st.rerun()
            else:
                st.error("Invalid Student ID. Try again.")

    if "student_id" not in st.session_state:
        _modal()
        st.stop()

# ----------------- LLM / TOPIC HELPERS -----------------
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY)

def create_chat_session():
    cfg = types.GenerateContentConfig(system_instruction=(
        "You are a friendly tutor. After each explanation ask an open-ended question "
        "to check understanding and encourage deeper thinking."
    ))
    return get_gemini_client().chats.create(model=MODEL_NAME, config=cfg)

def topic_detect(student_id: str, text: str):
    """Async fire-and-forget POST to HF Space."""
    try:
        requests.post(
            HF_SPACE_URL,
            json={"student_id": student_id, "text": text},
            timeout=25
        )
    except Exception as e:
        # Log quietly; we don't want to break UX
        print("Topic detection call failed:", e)

# ----------------- CHAT STATE -----------------
def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.setdefault("chats", {})
    st.session_state["chats"][cid] = {
        "title": "New Chat",
        "chat": create_chat_session(),
        "messages": []
    }
    st.session_state["active_chat"] = cid
    return cid

# ----------------- UI FLOW -----------------
st.set_page_config(page_title="School Tutor", layout="wide")

# 1. Login
login_modal()

# 2. Header
st.title("📚 School Tutor")
st.caption(f"Logged in as **{st.session_state['student_id']}**")

# 3. Sidebar – chat management
with st.sidebar:
    st.header("Chats")
    if st.button("➕ New Chat"):
        new_chat()

    # list existing
    chat_ids = list(st.session_state.get("chats", {}))
    if not chat_ids:
        new_chat()
        chat_ids = list(st.session_state["chats"])

    labels = [
        st.session_state["chats"][cid]["title"] for cid in chat_ids
    ]
    idx = chat_ids.index(st.session_state.get("active_chat", chat_ids[0]))
    sel = st.selectbox("Select chat", options=labels, index=idx)
    active_chat_id = chat_ids[labels.index(sel)]

    if st.button("🗑️ Delete selected"):
        st.session_state["chats"].pop(active_chat_id, None)
        if st.session_state["chats"]:
            st.session_state["active_chat"] = next(iter(st.session_state["chats"]))
        else:
            new_chat()
        st.rerun()

# 4. Main chat window
chat = st.session_state["chats"][active_chat_id]

# Display messages
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User input
prompt = st.chat_input("Ask me anything...")
if prompt:
    chat["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Fire topic detection in background
    topic_detect(st.session_state["student_id"], prompt)

    # Stream response
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

        # Rename chat if still default
        if chat["title"] == "New Chat":
            chat["title"] = prompt[:40].replace("\n", " ")
    except Exception as e:
        st.error(f"LLM error: {e}")
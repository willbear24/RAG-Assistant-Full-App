import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Longer than the backend's Ollama timeout so the API's own error reaches the user
ASK_TIMEOUT = float(os.environ.get("ASK_TIMEOUT", "150"))

st.set_page_config(page_title="RAG API Assistant", page_icon="🤖", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 RAG API Assistant")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API Connected")
        st.write(f"Ollama: {health.get('ollama', 'unknown')}")
        st.metric("Documents", health.get("documents", 0))
    except (requests.RequestException, ValueError):
        st.error("API not available")

    if st.button("🔄 Re-index Documents"):
        try:
            r = requests.post(f"{API_URL}/ingest", timeout=60)
            r.raise_for_status()
            st.success(r.json().get("message", "Done"))
        except requests.RequestException:
            st.error("Ingestion failed")
        else:
            st.rerun()

# --- Chat Interface ---
st.title("Ask Your Documents")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"{s['source']} (distance: {s['distance']:.3f})")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            r = requests.post(
                f"{API_URL}/ask", json={"question": prompt}, timeout=ASK_TIMEOUT
            )
            data = r.json()
            if not r.ok:
                detail = (
                    data.get("detail", "Unknown error")
                    if isinstance(data, dict)
                    else r.text
                )
                st.error(f"Error: {detail}")
                st.stop()
            st.write(data["answer"])

            sources = data.get("sources", [])
            if sources:
                with st.expander(
                    f"Sources ({data.get('confidence', 'unknown')} confidence)"
                ):
                    for s in sources:
                        st.caption(f"{s['source']} (dist: {s['distance']:.3f})")
                        st.write(s["text"][:200] + "...")

            st.session_state["messages"].append(
                {"role": "assistant", "content": data["answer"], "sources": sources}
            )
        except requests.Timeout:
            st.error(
                "The request timed out. The model may still be loading; please try again."
            )
        except requests.ConnectionError:
            st.error("Could not reach the API. Is the backend running?")
        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            st.error(f"Error: {e!s}")

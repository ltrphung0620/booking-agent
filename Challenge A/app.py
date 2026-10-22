"""Local UI: python -m streamlit run 'Challenge A/app.py' --server.port 8502."""

import os

import bootstrap
import streamlit as st
from dotenv import load_dotenv

from agent import MemoryChatSession
from memory_store import MemoryFileError, ProjectMemoryStore
from models import get_base_url


def main():
    load_dotenv(bootstrap.HERE / ".env", override=False)
    load_dotenv(bootstrap.DEMO / ".env", override=False)
    st.set_page_config(page_title="Challenge A · Booking Agent", page_icon="🧠", layout="centered")
    store = ProjectMemoryStore(os.environ.get("CHALLENGE_A_MEMORY_PATH") or
                               bootstrap.HERE / "data" / "project_memory.json")
    try:
        active = store.snapshot()
        history = store.history()
        memory_error = ""
    except (MemoryFileError, OSError):
        active, history = {"revision": 0, "active_rules": []}, {"records": []}
        memory_error = "Không đọc được file memory. Hãy kiểm tra file; dữ liệu đang được giữ nguyên."

    model_name = os.environ.get("FLIGHT_MODEL", "").strip()
    configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    title, actions = st.columns([3, 2], vertical_alignment="center")
    with title:
        st.title("Booking Agent")
    with actions:
        if st.button("＋ Cuộc chat mới", use_container_width=True):
            st.session_state.pop("challenge_a_chat", None)
            st.rerun()
    if memory_error:
        st.error(memory_error)
    if not configured or not model_name:
        st.info("Điền OPENAI_API_KEY, OPENAI_BASE_URL và FLIGHT_MODEL trong demo/.env rồi khởi động lại.")

    session = st.session_state.get("challenge_a_chat")
    if session is None and not memory_error:
        session = MemoryChatSession(model_name, store, base_url=get_base_url())
        st.session_state.challenge_a_chat = session

    chat_tab, memory_tab = st.tabs(["Hội thoại", "Bộ nhớ & bằng chứng"])
    with memory_tab:
        st.subheader("Quy ước hiện hành trong file")
        st.json(active)
        st.caption(f"File: {store.path}")
        st.subheader("Memory thực sự đã đưa vào context")
        if session and session.injections:
            st.caption("Snapshot dưới đây là nội dung được nối vào system message trước khi gọi model.")
            for evidence in session.turns[-1].get("memory_injections", []):
                with st.expander(f"Lượt gọi model {evidence['model_call']} · revision "
                                 f"{evidence['snapshot']['revision']}", expanded=True):
                    st.code(evidence["injected_context"], language="text")
        else:
            st.info("Phiên này chưa gọi model. Memory đã có thể đọc từ file, nhưng chưa được gửi vào context.")
        with st.expander("Lịch sử thay đổi · gồm các quy ước đã bị thay thế"):
            st.json(history)
        if session and session.turns:
            with st.expander("Các bước agent đã thực hiện"):
                for index, turn in enumerate(session.turns, 1):
                    tool_events = [e for e in turn["trace"] if e["event"] == "tool"]
                    if tool_events:
                        st.caption(f"Tin nhắn {index}")
                        for event in tool_events:
                            st.write(f"**{event['name']}** · {event['result']['status']}")
                            st.json({"args": event["args"], "result": event["result"]})

    with chat_tab:
        if not session or not session.turns:
            with st.chat_message("assistant"):
                st.write("Chào bạn! Mình hỗ trợ đặt vé và có thể ghi nhớ quy ước dự án qua các cuộc chat.")
        if session:
            for turn in session.turns:
                with st.chat_message("user"):
                    st.markdown(turn["user"])
                with st.chat_message("assistant"):
                    if turn["error"]:
                        st.error(turn["assistant"])
                    else:
                        st.markdown(turn["assistant"])

    if text := st.chat_input("Nhắn yêu cầu đặt vé hoặc quy ước cần ghi nhớ…", max_chars=6000,
                             disabled=not configured or not model_name or bool(memory_error)):
        with st.spinner("Agent đang xử lý…"):
            # All payments are simulated; the agent still requires a user request.
            session.respond(text, allow_payment=True)
        st.rerun()


if __name__ == "__main__":
    main()

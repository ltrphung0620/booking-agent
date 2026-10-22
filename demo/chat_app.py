"""Run: python -m streamlit run chat_app.py --server.address 127.0.0.1"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chat_agent import ChatSession
from mock_tools import SCENARIOS
from models import get_base_url


def main():
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    st.set_page_config(page_title="Booking Agent", page_icon="✈️", layout="centered")

    with st.sidebar:
        st.title("✈️ Booking Agent")
        st.caption("OpenAI / 9router · Trợ lý đặt vé")
        base_url = st.text_input("API Base URL", value=get_base_url(),
                                 help="Endpoint API, thường kết thúc bằng /v1. Không thêm /chat/completions.")
        model_name = st.text_input("Model ID", value=os.environ.get("FLIGHT_MODEL", ""),
                                   placeholder="Model ID chính xác từ nhà cung cấp/router")
        configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        if configured:
            st.success("Đã có API key trên máy chủ")
        else:
            st.warning("Chưa cấu hình OPENAI_API_KEY")
        scenario = st.selectbox("Kịch bản dữ liệu", SCENARIOS)
        allow_payment = st.checkbox("Cho phép thanh toán mô phỏng", value=False)
        if st.button("＋ Cuộc chat mới", use_container_width=True):
            st.session_state.pop("booking_chat", None)
            st.rerun()
        st.divider()
        st.caption("LLM gọi API thật tại endpoint đã chọn khi gửi tin nhắn. Vé và thanh toán là mô phỏng.")
        st.caption("Lịch sử chỉ giữ trong cuộc chat hiện tại. Persistent memory sẽ thêm ở Challenge A.")

    st.title("Hôm nay bạn muốn bay đi đâu?")
    st.caption("Trao đổi với trợ lý để tìm chuyến bay, giữ chỗ và kiểm tra booking.")
    if not configured or not model_name.strip():
        st.info("Để bắt đầu, sao chép demo/.env.example thành demo/.env, điền "
                "OPENAI_API_KEY, OPENAI_BASE_URL và FLIGHT_MODEL rồi khởi động lại ứng dụng. "
                "Với 9router, dùng key và model ID từ dashboard. "
                "Bạn cũng có thể đổi endpoint và model ở thanh bên.")

    session = st.session_state.get("booking_chat")
    # Changes to scenario need a new service; don't silently discard bookings.
    settings_changed = session is not None and (
        session.model_name != model_name.strip()
        or getattr(session, "base_url", None) != get_base_url(base_url)
        or session.harness.service.scenario != scenario)
    if settings_changed:
        st.warning("Bạn đã đổi endpoint, model hoặc kịch bản. Nhấn ‘Cuộc chat mới’ để áp dụng.")

    if session is None or not session.turns:
        with st.chat_message("assistant"):
            st.markdown("Chào bạn! Mình có thể tìm chuyến bay và hỗ trợ đặt vé. "
                        "Bạn muốn đi từ đâu, đến đâu, vào ngày nào và ngân sách bao nhiêu?")
        with st.expander("Dữ liệu và câu hỏi để thử"):
            st.write("Bài tập có dữ liệu cố định SGN → DAD ngày 07/10/2026; "
                     "đây là dữ liệu mô phỏng, kể cả khi ngày này đã qua.")
            st.code("Tìm vé SGN đi DAD ngày 2026-10-07, trước 12 giờ, ngân sách 2 triệu.\n"
                    "Có những lựa chọn nào?\nGiữ chỗ chuyến VN122 giúp mình.\n"
                    "Thanh toán booking đó nhé.\nKiểm tra lại booking của mình.", language=None)

    if session:
        for turn in session.turns:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                if turn["error"]:
                    st.error(turn["assistant"])
                else:
                    st.markdown(turn["assistant"])
                tool_events = [e for e in turn["trace"] if e["event"] == "tool"]
                if tool_events:
                    with st.expander("Các bước agent đã thực hiện"):
                        for event in tool_events:
                            st.write(f"**{event['name']}** · {event['result']['status']}")
                            st.json({"args": event["args"], "result": event["result"]})
        with st.sidebar:
            st.caption(f"Token được API báo cáo: {session.token_usage()['total_tokens']:,}")
            if session.harness.trip_configured:
                with st.expander("Thông tin chuyến đang tìm"):
                    st.json(session.harness.constraints.model_dump(mode="json"))
            if session.harness.service.bookings:
                with st.expander("Booking hiện tại", expanded=True):
                    st.json(list(session.harness.service.bookings.values()))

    text = st.chat_input("Nhắn cho trợ lý đặt vé…",
                         disabled=not configured or not model_name.strip() or settings_changed,
                         max_chars=6000)
    if text:
        if session is None:
            session = ChatSession(model_name, scenario, base_url=base_url)
            st.session_state.booking_chat = session
        with st.chat_message("user"):
            st.markdown(text)
        with st.spinner("Trợ lý đang xử lý…"):
            session.respond(text, allow_payment=allow_payment)
        st.rerun()


if __name__ == "__main__":
    main()

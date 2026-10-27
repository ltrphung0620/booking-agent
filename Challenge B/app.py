"""Local UI: python -m streamlit run 'Challenge B/app.py' --server.port 8503."""

import os
from pathlib import Path

import bootstrap
import streamlit as st
from dotenv import load_dotenv

from compaction import CompactionPolicy
from investigation import InvestigationSession
from log_fixtures import generate_logs
from models import get_base_url


def data_root() -> Path:
    root = Path(os.environ.get("CHALLENGE_B_DATA") or bootstrap.HERE / "data")
    if not (root / "logs").is_dir():
        generate_logs(root)  # Seeded, so every machine sees the same incident.
    return root


def new_session(settings: dict, model_name: str) -> InvestigationSession:
    policy = CompactionPolicy(budget_tokens=settings["budget"]) if settings["compaction"] else None
    return InvestigationSession(model_name, data_root(), mode=settings["mode"], compaction=policy,
                                base_url=get_base_url())


def context_tab(session: InvestigationSession):
    summary = session.usage_summary()
    cols = st.columns(4)
    cols[0].metric("Lần gọi LLM", summary["llm_calls"])
    cols[1].metric("Tổng input", f"{summary['total_input']:,}")
    cols[2].metric("Input lớn nhất", f"{summary['peak_input']:,}")
    cols[3].metric("Context hiện tại", f"{session.context_tokens():,}")
    st.caption(f"Nguồn số liệu: {summary['source']} (usage = số router trả về, estimate = ước lượng).")
    if session.context_log:
        st.subheader("Input tokens theo từng lần gọi model")
        st.line_chart({"input tokens": [e["input_tokens"] if e["input_tokens"] is not None
                                        else e["approx_tokens"] for e in session.context_log]})
    events = [e for t in session.turns for e in t["trace"]
              if e["event"] in ("compaction", "compaction_failed", "compaction_skipped", "prune")]
    st.subheader("Compact và prune")
    if events:
        for e in events:
            label = {"compaction": "Compact", "compaction_failed": "Compact lỗi",
                     "compaction_skipped": "Bỏ qua compact", "prune": "Prune tool output"}[e["event"]]
            detail = (f"{e['tokens_before']:,} → {e['tokens_after']:,} token" if "tokens_after" in e
                      else e.get("reason", ""))
            st.write(f"**{label}** · {detail}")
    else:
        st.info("Chưa có lần compact hay prune nào.")
    st.subheader("Checkpoint hiện tại")
    if session.checkpoint:
        st.json(session.checkpoint.model_dump())
    else:
        st.info("Chưa có checkpoint.")
    runtime = [e for t in session.turns for e in t["trace"] if e["event"] == "runtime_tool"]
    if runtime:
        with st.expander(f"Lời gọi tools.* trong runtime ({len(runtime)})"):
            st.caption("Trace chỉ ghi kích thước kết quả; nội dung log không rời runtime.")
            st.dataframe([{"name": e["name"], "args": str(e["args"]), "status": e["status"],
                           "tokens": e["tokens"]} for e in runtime], use_container_width=True)


def main():
    load_dotenv(bootstrap.HERE / ".env", override=False)
    load_dotenv(bootstrap.DEMO / ".env", override=False)
    st.set_page_config(page_title="Challenge B · Log Investigation", page_icon="🔎", layout="centered")
    model_name = os.environ.get("FLIGHT_MODEL", "").strip()
    configured = bool(os.environ.get("OPENAI_API_KEY", "").strip()) and bool(model_name)

    with st.sidebar:
        st.header("Cấu hình")
        mode = st.radio("Chế độ", ["native", "code"], horizontal=True,
                        help="native: read_file từng file. code: run_code, log thô không vào context.")
        compaction = st.toggle("Tự compact", value=True)
        budget = st.number_input("Budget (token)", 4_000, 200_000, 16_000, step=2_000,
                                 disabled=not compaction)
        st.caption("Đổi cấu hình sẽ bắt đầu cuộc chat mới.")
    settings = {"mode": mode, "compaction": compaction, "budget": int(budget)}

    title, new_chat, compact = st.columns([3, 2, 2], vertical_alignment="center")
    with title:
        st.title("Log Agent")
    if new_chat.button("＋ Cuộc chat mới", use_container_width=True):
        st.session_state.pop("challenge_b_chat", None)
    if not configured:
        st.info("Điền OPENAI_API_KEY, OPENAI_BASE_URL và FLIGHT_MODEL trong demo/.env rồi khởi động lại.")

    session = st.session_state.get("challenge_b_chat")
    if session is None or st.session_state.get("challenge_b_settings") != settings:
        session = new_session(settings, model_name)
        st.session_state.challenge_b_chat = session
        st.session_state.challenge_b_settings = settings
    if compact.button("Compact ngay", use_container_width=True, disabled=not configured):
        result = session.maybe_compact(force=True)
        if result["status"] == "ok":
            st.toast(f"Đã compact: {result['tokens_before']:,} → {result['tokens_after']:,} token")
        else:
            st.toast(result["reason"])

    chat_tab, context = st.tabs(["Hội thoại", "Context & token"])
    with context:
        context_tab(session)
    with chat_tab:
        if not session.turns:
            with st.chat_message("assistant"):
                st.write("Chào bạn! Mình điều tra log tháng 10/2026 của hệ thống đặt vé. "
                         "Bạn muốn tìm hiểu lỗi gì?")
        for turn in session.turns:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                (st.error if turn["error"] else st.markdown)(turn["assistant"])

    if text := st.chat_input("Hỏi về log, ví dụ: mã lỗi nào nhiều nhất trong tháng 10?",
                             max_chars=6000, disabled=not configured):
        with st.spinner("Agent đang điều tra…"):
            session.respond(text)
        st.rerun()


if __name__ == "__main__":
    main()

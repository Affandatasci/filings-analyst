"""Step 4.4: the chat page, and the frame of the app (it also holds the second page, app\\dashboard.py).

    streamlit run app\\streamlit_app.py

Keys come from the Streamlit Secrets box first and from .env second (config.py does that). Every question goes through
monitor.flow.ask: the limits, the guards, the trace and the log. The role switch is a demo, not a login.
"""
import sys
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Filings Analyst", page_icon="📄", layout="wide")  # must be the first Streamlit call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  (after streamlit, so the Secrets box is read first)
from monitor import flow, store  # noqa: E402
from security import access  # noqa: E402

REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
ROLE_WORDS = {
    "guest": "guest: the public filings only",
    "analyst": "analyst: public and internal files",
    "hr": "hr: public, internal and HR files (made-up company data)",
}
MODE_WORDS = {"plain": "Plain: search, then one answer", "agent": "Agent: it can search again, read a page and calculate (slower, costs more)"}
EXAMPLE = "What drove the change in operating margin for 3M in FY2022?"


def missing_keys():
    return [name for name in REQUIRED_KEYS if not getattr(config, name, None)]


def make_turn(question, role, mode, outcome):
    """What the page needs from one finished question, as plain data that can sit in the session."""
    result = outcome.result
    turn = {"question": question, "role": role, "mode": mode, "request_id": outcome.request_id, "rated": None}
    if result is None:  # a limit said no: nothing was asked, nothing was spent
        return {**turn, "stopped": True, "text": outcome.limit.text}
    return {
        **turn, "stopped": False, "text": result.answer or "The bot gave no answer.", "blocked": result.blocked, "citations": result.citations,
        "pieces": [{"label": f"{hit.doc}, page {hit.page}", "score": hit.score, "text": hit.text[:900]} for hit in result.hits],
        "cents": result.cost_cents, "seconds": result.seconds, "steps": sum(1 for event in result.events if event.get("stage") == "agent"),
    }


def rate(index, value):
    turn = st.session_state.history[index]
    turn["rated"] = value if store.set_thumb(turn["request_id"], value) else 0  # 0 means: the rating could not be saved


def show_turn(turn, index):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn["stopped"]:
            st.warning(turn["text"])
            return
        if str(turn["blocked"]).startswith("error:"):
            st.error(turn["text"])
        elif turn["blocked"]:
            st.warning(turn["text"])
            st.caption(f"Stopped by a safety check ({turn['blocked']}).")
        else:
            st.markdown(turn["text"])
        if turn["citations"]:
            st.caption("Cited: " + "; ".join(f"{item.get('doc')} page {item.get('page')}" for item in turn["citations"]))
        steps = f", {turn['steps']} tool steps" if turn["mode"] == "agent" else ""
        st.caption(f"{turn['mode']} mode, role {turn['role']}{steps}. {turn['cents']:.3f} cents and {turn['seconds']:.1f} seconds.")
        if turn["pieces"]:
            with st.expander(f"The {len(turn['pieces'])} pieces of the filings the bot was given"):
                for piece in turn["pieces"]:
                    st.markdown(f"**{piece['label']}** (score {piece['score']:.3f})")
                    st.text(piece["text"])
        if turn["request_id"] is not None and turn["rated"] is None:
            good, bad, _ = st.columns([1, 1, 6])
            good.button("Good answer", key=f"good{index}", on_click=rate, args=(index, 1))
            bad.button("Bad answer", key=f"bad{index}", on_click=rate, args=(index, -1))
        elif turn["rated"]:
            st.caption("Thank you, your rating is saved.")
        elif turn["rated"] == 0:
            st.caption("Your rating could not be saved.")


def chat_page():
    state = st.session_state
    state.setdefault("session_id", uuid.uuid4().hex[:12])  # one visit; not a person
    state.setdefault("asked", 0)
    state.setdefault("history", [])

    st.title("Filings Analyst")
    st.write("Ask questions about 37 real company filings from the SEC. Every answer shows the pages it used.")
    st.info(config.DEMO_NOTICE)
    st.caption("Questions and answers are logged to find and fix mistakes. Do not type personal data.")

    missing = missing_keys()
    if missing:
        st.error("These settings are missing: " + ", ".join(missing) + ". Add them to the app Secrets (online) or to .env (laptop).")
        st.stop()

    with st.sidebar:
        role = st.selectbox("Role", list(access.ROLES), format_func=ROLE_WORDS.get, help="A demo switch: the role decides which files the search may read.")
        mode = st.radio("Mode", list(MODE_WORDS), format_func=MODE_WORDS.get)
        st.write(f"Questions left in this visit: {max(config.DEMO_SESSION_QUESTIONS - state.asked, 0)}")
        if isinstance(store.get_client(), store.FileClient):
            st.caption("No Supabase keys: the log stays on this computer.")

    for index, turn in enumerate(state.history):
        show_turn(turn, index)
    if not state.history:
        st.caption(f"For example: {EXAMPLE}")

    question = st.chat_input(f"Ask about a filing (up to {config.MAX_QUESTION_CHARS} characters)")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"), st.spinner("Searching the filings (the first question takes about a minute while the model loads) ..."):
            outcome = flow.ask(question, role, mode, session=state.session_id, asked=state.asked)
        state.asked += outcome.result is not None
        state.history.append(make_turn(question, role, mode, outcome))
        st.rerun()


def main():
    pages = [
        st.Page(chat_page, title="Ask the filings", icon="💬", default=True),
        st.Page(str(Path(__file__).with_name("dashboard.py")), title="Scores and monitor", icon="📊"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":  # a Streamlit page runs as __main__; an import (the tests) does not start the page
    main()

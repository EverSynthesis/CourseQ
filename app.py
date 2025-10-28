# app.py
import os
import json
import random
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
from openai import OpenAI, APIStatusError, RateLimitError

# ---------- App config ----------
st.set_page_config(page_title="Question Randomizer", page_icon="🎲")
st.title("🎲 Course Question Randomizer")

# ---------- OpenAI client ----------
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a precise extraction assistant. "
    "You will read an attached PDF that contains multiple course questions. "
    "Extract ONLY the question prompts, not answers or solutions. "
    "If questions have subparts (a), (b), (c), keep them attached to the same question text. "
    "Ignore answer keys, marking schemes, prefaces, or unrelated text. "
    "Return STRICT JSON with the following schema (no extra commentary):\n\n"
    "{\n"
    '  "questions": [\n'
    '    {\n'
    '      "index": <1-based integer>,\n'
    '      "question_text": "<full question prompt as a single string>",\n'
    '      "page_range": "<e.g., 3 or 3-4 if known, else omit or null>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Ensure all newlines in question_text are preserved where helpful, but keep valid JSON."
)

def extract_questions_from_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], str]:
    """Uploads PDF to OpenAI Files, asks model to extract questions as strict JSON."""
    pdf_io = BytesIO(file_bytes)
    pdf_io.name = filename
    up = client.files.create(file=pdf_io, purpose="user_data")

    resp = client.responses.create(
        model="gpt-5-mini",  # switch to "gpt-5" for higher quality
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "input_file", "file_id": up.id},
                {"type": "input_text", "text": "Extract the questions per the instructions."}
            ]}
        ],
    )
    text = resp.output_text.strip()

    def try_parse_json(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
            raise

    data = try_parse_json(text)
    questions = data.get("questions", [])

    # Normalize indices & text
    for i, q in enumerate(questions, start=1):
        if not isinstance(q.get("index"), int):
            q["index"] = i
        q["question_text"] = str(q.get("question_text", "")).strip() or "(empty question text)"
    return questions, text


# ---------- Persistence (Supabase) ----------
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
sb: "Client | None" = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if (create_client and SUPABASE_URL and SUPABASE_KEY) else None
)

def serialize_store(store: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sets -> lists so JSON is valid."""
    out = {}
    for subj, payload in store.items():
        out[subj] = {
            "questions": payload.get("questions", []),
            "used": sorted(list(payload.get("used", set()))),
            "raw_log": payload.get("raw_log", []),
        }
    return out

def deserialize_store(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists -> sets; ensure structure."""
    out = {}
    for subj, p in (payload or {}).items():
        used_list = p.get("used", [])
        out[subj] = {
            "questions": p.get("questions", []),
            "used": set(used_list if isinstance(used_list, list) else []),
            "raw_log": p.get("raw_log", []),
        }
    return out

def load_store(user_id: str) -> Dict[str, Any]:
    if not (sb and user_id): return {}
    res = sb.table("subject_store").select("payload").eq("user_id", user_id).execute()
    return deserialize_store(res.data[0]["payload"]) if res.data else {}

def save_store(user_id: str, store: Dict[str, Any]) -> None:
    if not (sb and user_id): return
    sb.table("subject_store").upsert({
        "user_id": user_id,
        "payload": serialize_store(store)
    }).execute()

# ---------- Session state ----------
if "subjects" not in st.session_state:
    # subjects: { name: {"questions":[...], "used": set(indices), "raw_log":[raw_json_str,...]} }
    st.session_state.subjects = {}
if "current_subject" not in st.session_state:
    st.session_state.current_subject = None
if "loaded_from_db" not in st.session_state:
    st.session_state.loaded_from_db = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

# ---------- Sidebar: identity + subject mgmt ----------
with st.sidebar:
    st.header("👤 User")
    st.session_state.user_id = st.text_input(
        "Your name/email (to save your data):",
        value=st.session_state.user_id,
        placeholder="e.g., alice@nus.edu.sg"
    ).strip()

    if not st.session_state.loaded_from_db and st.session_state.user_id:
        # initial load for this user
        try:
            st.session_state.subjects = load_store(st.session_state.user_id) or {}
        except Exception as e:
            st.warning("Could not load your saved subjects yet.")
        st.session_state.loaded_from_db = True

    st.header("📚 Subjects")
    if "new_subject_name" not in st.session_state:
        st.session_state.new_subject_name = ""

    def persist_now():
        if st.session_state.user_id:
            save_store(st.session_state.user_id, st.session_state.subjects)

    # Callbacks
    def create_subject_cb():
        name = (st.session_state.new_subject_name or "").strip()
        if not name:
            st.session_state._flash = ("error", "Please enter a subject name."); return
        if name in st.session_state.subjects:
            st.session_state._flash = ("warning", "That subject already exists."); return
        st.session_state.subjects[name] = {"questions": [], "used": set(), "raw_log": []}
        st.session_state.current_subject = name
        st.session_state.new_subject_name = ""
        persist_now()
        st.rerun()

    def delete_current_subject_cb():
        cur = st.session_state.current_subject
        if cur and cur in st.session_state.subjects:
            del st.session_state.subjects[cur]
            st.session_state.current_subject = None
            st.session_state._flash = ("success", "Subject deleted.")
            persist_now()
            st.rerun()

    # Picker
    subject_names = sorted(st.session_state.subjects.keys())
    options = ["(none)"] + subject_names
    default = st.session_state.current_subject if st.session_state.current_subject in subject_names else "(none)"
    choice = st.selectbox("Select subject", options=options, index=options.index(default), help="Pick a subject to work on.")
    st.session_state.current_subject = None if choice == "(none)" else choice

    st.divider()
    st.caption("Create a new subject")
    st.text_input("New subject name", key="new_subject_name",
                  placeholder="e.g., Calculus, Signals, EE2028",
                  help="Type a name, then click Create Subject.")
    st.button("➕ Create Subject", use_container_width=True, on_click=create_subject_cb)

    if st.session_state.current_subject:
        st.write("")  # spacer
        st.button("🗑️ Delete Current Subject", use_container_width=True, type="secondary", on_click=delete_current_subject_cb)

    # Manual Save button (optional)
    if st.button("💾 Save now", use_container_width=True, disabled=not st.session_state.user_id):
        persist_now()
        st.success("Saved.")

    # Flash messages
    if "_flash" in st.session_state:
        level, msg = st.session_state._flash
        {"error": st.error, "warning": st.warning}.get(level, st.success)(msg)
        del st.session_state._flash

# ---------- Main: Upload, Extract, Randomize ----------
st.markdown("Upload a PDF containing questions and extract them **into the selected subject**. "
            "Then draw a random practice question from that subject.")

if st.session_state.current_subject is None:
    st.warning("Select or create a subject from the sidebar to begin.")
else:
    subj = st.session_state.current_subject
    store = st.session_state.subjects[subj]

    uploaded = st.file_uploader(f"Upload PDF to add questions to **{subj}**", type=["pdf"], key=f"uploader_{subj}")

    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        extract_btn = st.button("📥 Extract", use_container_width=True, disabled=(uploaded is None))
    with colB:
        random_btn = st.button("🎲 Randomize one", use_container_width=True, disabled=(len(store["questions"]) == 0))
    with colC:
        reset_btn = st.button("🔁 Reset draws", use_container_width=True, disabled=(len(store["used"]) == 0))
    with colD:
        clear_btn = st.button("🧹 Clear questions", use_container_width=True, disabled=(len(store["questions"]) == 0))

    if extract_btn and uploaded is not None:
        with st.spinner(f"Extracting questions into {subj}…"):
            try:
                qs, raw = extract_questions_from_pdf(uploaded.read(), uploaded.name)
                if not qs:
                    st.warning("I didn't find any questions. Try another PDF or check its formatting.")
                else:
                    offset = len(store["questions"])
                    for i, q in enumerate(qs, start=1):
                        store["questions"].append({
                            "index": offset + i,
                            "question_text": q.get("question_text", ""),
                            "page_range": q.get("page_range")
                        })
                    store["raw_log"].append(raw)
                    st.success(f"Added {len(qs)} question(s) to {subj}. Total now: {len(store['questions'])}.")
                    # persist
                    if st.session_state.user_id:
                        save_store(st.session_state.user_id, st.session_state.subjects)
            except RateLimitError as e:
                st.error("Rate/quota limit—check billing/usage."); st.code(str(e))
            except APIStatusError as e:
                st.error(f"API error {e.status_code}."); st.code(e.message)
            except Exception as e:
                st.error("Unexpected error during extraction."); st.code(repr(e))

    if random_btn and store["questions"]:
        all_idxs = list(range(len(store["questions"])))
        unused = [i for i in all_idxs if i not in store["used"]]
        if not unused:
            st.info("All questions have been drawn once. Click **Reset draws** to start over.")
        else:
            pick = random.choice(unused)
            store["used"].add(pick)
            q = store["questions"][pick]
            st.subheader(f"🎯 Your practice question (Subject: {subj})")
            st.markdown(f"**Q{q['index']}**" + (f"  _(pages {q['page_range']})_" if q.get("page_range") else ""))
            st.text(q["question_text"])
            # persist draw state
            if st.session_state.user_id:
                save_store(st.session_state.user_id, st.session_state.subjects)

    if reset_btn:
        store["used"] = set()
        st.success("Draw history reset for this subject.")
        if st.session_state.user_id:
            save_store(st.session_state.user_id, st.session_state.subjects)

    if clear_btn:
        store["questions"] = []
        store["used"] = set()
        store["raw_log"] = []
        st.success("Cleared all questions for this subject.")
        if st.session_state.user_id:
            save_store(st.session_state.user_id, st.session_state.subjects)

    if store["questions"]:
        with st.expander(f"📄 Preview questions in {subj} ({len(store['questions'])})"):
            for q in store["questions"]:
                st.markdown(f"**Q{q['index']}**" + (f"  _(pages {q['page_range']})_" if q.get("page_range") else ""))
                st.text(q["question_text"])
                st.markdown("---")

st.markdown("---")
st.caption("Tip: Each subject keeps its own question pool and draw history. "
           "Add more PDFs to grow a subject, or create new subjects for other modules.")

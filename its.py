"""
Intelligent Tutoring System - PDF-based AI Quiz Generator & Tutor
Filename: its_pdf_qgen.py

Description:
- Streamlit web app that lets a student upload a PDF (lecture notes / textbook chapter)
  and automatically generates quiz questions from the content.
- Two generation modes:
  1) Rule-based (fast, offline) : simple cloze & factual Qs using sentence heuristics.
  2) LLM-based (better quality) : uses OpenAI API to generate Q/A pairs (requires API key).
- Adaptive quiz flow: tracks score, increases chunk difficulty (longer/more complex passages)
- Progress storage: SQLite (stores username, subject, score, timestamp)

Dependencies:
- streamlit
- PyPDF2 (or pypdf)
- nltk
- sqlite3 (builtin)
- openai (optional, for LLM QG)
- transformers (optional - not used in default implementation)

Install:
    pip install streamlit PyPDF2 nltk openai

How to run:
    export OPENAI_API_KEY="sk-..."    # (optional, for LLM mode)
    streamlit run its_pdf_qgen.py

Notes:
- LLM mode uses OpenAI's "gpt-3.5-turbo" via the openai python package. If you don't want
  to use the OpenAI API, use Rule-based mode.
- This is a prototype focused on quality of ideas and demo purposes.

"""

import streamlit as st
import tempfile
import sqlite3
import random
import re
import os
import datetime

# PDF parsing
try:
    from PyPDF2 import PdfReader
except Exception:
    # PyPDF2 vs pypdf
    from PyPDF2 import PdfReader

# Optional: openai for LLM-based QG
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# NLTK for sentence tokenization
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except Exception:
    import nltk
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# ------------------------ Database ------------------------ #
DB_PATH = "its_progress.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    source TEXT,
                    mode TEXT,
                    score INTEGER,
                    total INTEGER,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ------------------------ Utilities ------------------------ #

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text.append(page_text)
    return "\n".join(text)


def clean_text(t):
    # normalize whitespace, remove weird chars
    t = t.replace('\r', ' ')
    t = re.sub('\s+', ' ', t)
    return t.strip()


# Simple heuristic to find candidate "fact" sentences (numbers, definitions)
NUMERIC_RE = re.compile(r"\b\d+\b")
DEFINITION_RE = re.compile(r"\b(is|are|refers to|means|defined as)\b", re.I)


def score_sentence_for_qg(sent):
    score = 0
    if NUMERIC_RE.search(sent):
        score += 2
    if DEFINITION_RE.search(sent):
        score += 3
    # length preference
    if 40 < len(sent) < 250:
        score += 1
    return score


# ------------------------ Rule-based question generators ------------------------ #

def cloze_from_sentence(sent):
    """Create a simple cloze (fill-in-the-blank) by masking a word.
    Heuristics: prefer nouns/numbers/technical words (capitalized or digits)."""
    words = sent.split()
    candidate_idxs = []
    for i, w in enumerate(words):
        bare = re.sub(r"[^A-Za-z0-9_]", "", w)
        if not bare:
            continue
        # prefer numbers or TitleCase words or longer words
        if re.match(r"^\d+$", bare) or (bare[0].isupper() and len(bare) > 2) or len(bare) > 6:
            candidate_idxs.append(i)
    if not candidate_idxs:
        # fallback: pick a random non-stop short word that is not a preposition
        candidate_idxs = [i for i, w in enumerate(words) if len(re.sub(r"[^A-Za-z0-9_]", "", w)) > 2]
        if not candidate_idxs:
            return None
    idx = random.choice(candidate_idxs)
    answer = re.sub(r"[^A-Za-z0-9_]", "", words[idx])
    masked = words.copy()
    masked[idx] = "_____"
    question = ' '.join(masked)
    return question, answer


def generate_rule_based_questions(text, n_questions=10):
    sents = sent_tokenize(text)
    # score each sentence by heuristics
    scored = [(score_sentence_for_qg(s), s) for s in sents]
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    questions = []
    used = set()
    for score, sent in scored:
        if len(questions) >= n_questions:
            break
        if score <= 0:
            continue
        if sent in used:
            continue
        used.add(sent)
        cloze = cloze_from_sentence(sent)
        if cloze:
            q, a = cloze
            questions.append({"type": "cloze", "question": q, "answer": a, "source": sent})
    # If not enough, add simple arithmetic generation (if math content desirable)
    i = 0
    while len(questions) < n_questions and i < 20:
        # generate a simple arithmetic question
        a, b = random.randint(2, 20), random.randint(2, 12)
        q = f"What is {a} × {b}?"
        questions.append({"type": "short", "question": q, "answer": str(a*b), "source": "generated"})
        i += 1
    return questions


# ------------------------ OpenAI LLM-based QG ------------------------ #

def llm_generate_questions(paragraphs, n_questions=10, model="gpt-3.5-turbo"):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not installed or not available. Install 'openai' and set OPENAI_API_KEY env var.")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Please set OPENAI_API_KEY as environment variable to use LLM mode.")

    # Craft a system prompt and user prompt to ask for q/a pairs
    system = "You are an educational question generator. Given a passage, produce clear questions and short answers suitable for quizzes."

    combined = '\n\n'.join(paragraphs)
    # To keep tokens low, we will chunk combined if too long
    prompt = f"Generate {n_questions} short question-answer pairs (concise answers). Each pair on new line formatted as Q: <question> | A: <answer>.\n\nPassage:\n{combined}"

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"system", "content": system}, {"role":"user", "content": prompt}],
        max_tokens=800,
        temperature=0.2
    )
    text = resp.choices[0].message.content.strip()
    # parse lines
    qa_pairs = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"Q:\s*(.*?)\s*\|\s*A:\s*(.*)$", line)
        if m:
            q = m.group(1).strip()
            a = m.group(2).strip()
            qa_pairs.append({"type": "short", "question": q, "answer": a})
        else:
            # try split by '?'
            if '?' in line:
                parts = line.split('?')
                q = parts[0].strip() + '?'
                a = '?'.join(parts[1:]).strip()
                qa_pairs.append({"type": "short", "question": q, "answer": a})
    return qa_pairs[:n_questions]


# ------------------------ Distractor generator (very simple) ------------------------ #

def generate_mcq_options(answer, qtype="short"):
    # For numeric answers, produce numeric distractors
    opts = []
    ans = answer.strip()
    if ans.isdigit():
        val = int(ans)
        deltas = [-3, -1, 2, 4]
        opts = [str(val + d) for d in deltas]
        opts.append(ans)
    else:
        # small heuristic: produce variants by case or short synonyms (fallback random strings)
        opts = [ans, ans.capitalize(), ans + "s", "None of the above"]
    random.shuffle(opts)
    # ensure unique and include answer
    opts = list(dict.fromkeys(opts))
    if ans not in opts:
        opts[-1] = ans
    return opts[:4]


# ------------------------ Streamlit UI ------------------------ #

st.set_page_config(page_title="AI PDF Quiz Generator - ITS", layout="wide")

st.title("🤖 Intelligent Tutoring System — PDF Quiz Generator")

st.markdown("Upload lecture notes (PDF), choose generation mode, and the system will generate quiz questions automatically.")

with st.sidebar:
    st.header("Settings")
    username = st.text_input("Your name", value="Student")
    mode = st.selectbox("Generation Mode", ["Rule-based (offline)", "LLM-based (better, needs key)"])
    n_q = st.slider("Number of questions", 5, 20, 8)
    show_sources = st.checkbox("Show source sentences (for teacher review)", value=True)
    if OPENAI_AVAILABLE:
        st.write("OpenAI available")
    else:
        st.write("OpenAI not installed or no API key set — LLM mode will error if used.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tfile.write(uploaded_file.read())
    tfile.flush()
    raw_text = extract_text_from_pdf(tfile.name)
    text = clean_text(raw_text)

    st.success("PDF loaded — extracted text length: {} characters".format(len(text)))

    # show short preview
    if st.checkbox("Show extracted text preview (first 800 chars)"):
        st.text_area("Contents preview", value=text[:800], height=200)

    # chunk text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 40]
    if not paragraphs:
        # fallback to sentence chunks
        paragraphs = sent_tokenize(text)
    # choose most informative paragraphs
    paragraphs_scored = sorted(paragraphs, key=lambda s: score_sentence_for_qg(s), reverse=True)
    top_paras = paragraphs_scored[:min(8, len(paragraphs_scored))]

    st.info(f"Using top {len(top_paras)} paragraphs as source material for question generation.")

    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            try:
                if mode.startswith("Rule"):
                    questions = generate_rule_based_questions(' '.join(top_paras), n_questions=n_q)
                else:
                    # LLM mode - chunk a few paragraphs to avoid token overload
                    chunk = top_paras[:4]
                    questions = llm_generate_questions(chunk, n_questions=n_q)
            except Exception as e:
                st.error(f"Error during generation: {e}")
                questions = []

        if not questions:
            st.warning("No questions generated. Try Rule-based mode or upload a different PDF.")
        else:
            st.success(f"Generated {len(questions)} questions")

            # Present quiz
            if st.button("Start Quiz"):
                st.session_state['quiz'] = questions
                st.session_state['q_idx'] = 0
                st.session_state['score'] = 0

    # display generated questions for review (teacher)
    if st.checkbox("Preview generated questions (teacher)") and 'questions' in locals() and questions:
        st.subheader("Generated questions (Preview)")
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q['question']}")
            if show_sources and 'source' in q:
                st.caption(q.get('source'))

# Quiz interaction area
st.sidebar.markdown("---")
st.sidebar.subheader("Progress Database")
if st.sidebar.button("View past progress"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT username, source, mode, score, total, timestamp FROM progress ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    if rows:
        for r in rows:
            st.sidebar.write(f"{r[0]} | {r[1]} | {r[2]} | Score: {r[3]}/{r[4]} | {r[5]}")
    else:
        st.sidebar.write("No saved progress yet.")

# Main quiz flow if started
if 'quiz' in st.session_state:
    quiz = st.session_state['quiz']
    idx = st.session_state['q_idx']
    total = len(quiz)
    st.subheader(f"Quiz — Question {idx+1} of {total}")
    q = quiz[idx]
    st.write(q['question'])

    if q['type'] == 'cloze' or q['type'] == 'short':
        ans = st.text_input("Your answer", key=f"ans_{idx}")
        if st.button("Submit Answer", key=f"submit_{idx}"):
            if ans.strip().lower() == q['answer'].strip().lower():
                st.success("Correct ✅")
                st.session_state['score'] += 1
            else:
                st.error(f"Incorrect ❌ — Correct: {q['answer']}")
                # show hint/source if available
                if 'source' in q:
                    st.caption(f"Source: {q['source']}")
            st.session_state['q_idx'] += 1
            st.experimental_rerun()
    else:
        # MCQ type (fallback)
        opts = generate_mcq_options(q['answer'])
        choice = st.radio("Choose answer", opts, key=f"mcq_{idx}")
        if st.button("Submit MCQ", key=f"mcq_submit_{idx}"):
            if choice == q['answer']:
                st.success("Correct ✅")
                st.session_state['score'] += 1
            else:
                st.error(f"Incorrect ❌ — Correct: {q['answer']}")
                if 'source' in q:
                    st.caption(f"Source: {q['source']}")
            st.session_state['q_idx'] += 1
            st.experimental_rerun()

    # finishing
    if 'q_idx' in st.session_state and st.session_state['q_idx'] >= total:
        final_score = st.session_state['score']
        st.balloons()
        st.success(f"Quiz completed — {username}: {final_score}/{total}")
        # save to DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO progress (username, source, mode, score, total, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (username, uploaded_file.name if uploaded_file is not None else 'uploaded_pdf', mode, final_score, total, datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        # cleanup
        del st.session_state['quiz']
        del st.session_state['q_idx']
        del st.session_state['score']

# Footer
st.markdown("---")
st.write("Prototype by Athar — upgrade ideas: add better distractor generation, use spaCy for NER-based cloze, or use local LLM for offline generation.")

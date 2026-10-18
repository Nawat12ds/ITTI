
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, session
from google import genai
from google.genai import types

# -------------------------
# 1. ตั้งค่าพื้นฐาน
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

app = Flask(__name__)
app.secret_key = SECRET_KEY or os.urandom(32).hex()

# เปลี่ยนชื่อโมเดลได้จากไฟล์ .env หากต้องการ
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# -------------------------
# 2. ฐานข้อมูล SQLite
# -------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                persona_name TEXT NOT NULL,
                persona_style TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def create_conversation(name, style):
    conversation_id = str(uuid.uuid4())

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO conversations
            (id, persona_name, persona_style, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                conversation_id,
                name,
                style,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    return conversation_id


def get_current_conversation():
    conversation_id = session.get("conversation_id")

    if conversation_id:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

        if row:
            return row

    name = session.get("persona_name", "ครู AI")
    style = session.get(
        "persona_style",
        "ใจเย็น อธิบายเป็นขั้นตอน และยกตัวอย่าง",
    )

    conversation_id = create_conversation(name, style)
    session["conversation_id"] = conversation_id

    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()


# -------------------------
# 3. ฟังก์ชันที่ AI เรียกใช้ได้
# -------------------------

def save_study_note(title: str, content: str) -> str:
    """Save a useful study note to the database.
    Use this when the user asks to save or remember study notes.
    """
    title = title.strip()[:150]
    content = content.strip()[:10000]

    if not title or not content:
        return "ไม่สามารถบันทึกได้ เพราะหัวข้อหรือเนื้อหาว่าง"

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO study_notes (title, content, created_at)
            VALUES (?, ?, ?)
            """,
            (title, content, datetime.now().isoformat(timespec="seconds")),
        )

    return f"บันทึกโน้ตเรื่อง '{title}' เรียบร้อยแล้ว"


def list_study_notes() -> str:
    """List study notes saved in the database.
    Use this when the user asks to view their saved study notes.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT title, content, created_at
            FROM study_notes
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()

    if not rows:
        return "ยังไม่มีโน้ตที่บันทึกไว้"

    result = []

    for row in rows:
        result.append(
            f"หัวข้อ: {row['title']}\n"
            f"เนื้อหา: {row['content']}\n"
            f"วันที่: {row['created_at']}"
        )

    return "\n\n".join(result)


# -------------------------
# 4. สร้างคำสั่งสำหรับ AI Tutor
# -------------------------

def build_system_instruction(persona_name, persona_style):
    return f"""
คุณคือ AI Tutor บนเว็บไซต์สำหรับช่วยผู้ใช้เรียนรู้ทุกวิชา

ชื่อหรือบุคลิกที่ผู้ใช้เลือก: {persona_name}
สไตล์การสอนที่ผู้ใช้ต้องการ: {persona_style}

กติกาการสอน:
1. สอนได้ทุกวิชา เช่น คณิตศาสตร์ วิทยาศาสตร์ ภาษา ภาษาไทย
   ภาษาอังกฤษ สังคม และการเขียนโปรแกรม
2. อธิบายให้เข้าใจง่าย เป็นขั้นตอน และเหมาะกับระดับของผู้เรียน
3. หากโจทย์ยาก ให้แบ่งเป็นขั้นตอนย่อย พร้อมยกตัวอย่าง
4. หากผู้ใช้เข้าใจผิด ให้แก้ไขอย่างสุภาพและอธิบายเหตุผล
5. หากข้อมูลในคำถามไม่พอ ให้ถามเพิ่มเติมแทนการเดา
6. หากไม่แน่ใจในข้อเท็จจริง ให้บอกตรง ๆ ว่าไม่แน่ใจ
7. ปรับน้ำเสียงให้ใกล้เคียงสไตล์ที่ผู้ใช้ระบุ
   แต่ห้ามกล่าวอ้างว่าคุณเป็นบุคคลจริงหรือตัวละครนั้นจริง ๆ
8. เมื่อผู้ใช้ขอให้บันทึกโน้ตการเรียน ให้ใช้ฟังก์ชัน
   save_study_note หากเหมาะสม
9. เมื่อผู้ใช้ขอดูโน้ตที่บันทึกไว้ ให้ใช้ฟังก์ชัน list_study_notes
10. อย่าเปิดเผย API key รหัสผ่าน หรือข้อมูลลับของระบบ
11. ตอบเป็นภาษาเดียวกับผู้ใช้ เว้นแต่ผู้ใช้ขอภาษาอื่น

เป้าหมายคือช่วยให้ผู้ใช้เข้าใจเนื้อหา ไม่ใช่แค่ให้คำตอบอย่างเดียว
"""


# -------------------------
# 5. ส่งคำถามไปยัง Gemini
# -------------------------

def ask_ai(user_message, conversation):
    if client is None:
        raise RuntimeError(
            "ยังไม่ได้ตั้งค่า GEMINI_API_KEY ในไฟล์ .env"
        )

    with get_db() as conn:
        history = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 12
            """,
            (conversation["id"],),
        ).fetchall()

    # เรียงข้อความจากเก่าไปใหม่
    history = list(reversed(history))

    contents = []

    for item in history:
        contents.append(
            types.Content(
                role=item["role"],
                parts=[types.Part.from_text(text=item["content"])],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(
                conversation["persona_name"],
                conversation["persona_style"],
            ),
            tools=[save_study_note, list_study_notes],
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )

    answer = response.text

    if not answer:
        answer = (
            "ขอโทษครับ รอบนี้ AI ยังสร้างคำตอบไม่ได้ "
            "ลองถามใหม่อีกครั้งนะ"
        )

    # บันทึกบทสนทนาเมื่อเรียก AI สำเร็จ
    with get_db() as conn:
        now = datetime.now().isoformat(timespec="seconds")

        conn.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation["id"], "user", user_message, now),
        )

        conn.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation["id"], "model", answer, now),
        )

    return answer


# -------------------------
# 6. หน้าเว็บไซต์
# -------------------------


PAGE = r"""
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NEURA — AI Learning System</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #050814;
    --panel: rgba(11, 20, 43, .88);
    --panel2: #0b1530;
    --line: rgba(55, 153, 255, .24);
    --cyan: #38e8ff;
    --blue: #467aff;
    --purple: #a855f7;
    --text: #eaf3ff;
    --muted: #8295b8;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(ellipse at 48% -20%, #172b60 0%, transparent 48%),
        radial-gradient(ellipse at 100% 80%, #25104b 0%, transparent 35%),
        var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans Thai', sans-serif;
}

button, input, textarea { font: inherit; }
button { cursor: pointer; }

.app {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 225px minmax(0, 1fr);
}

.sidebar {
    padding: 28px 16px;
    border-right: 1px solid var(--line);
    background: rgba(4, 8, 23, .84);
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 8px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    color: var(--cyan);
    font-size: 25px;
    background: linear-gradient(135deg, #123d70, #28175c);
    border: 1px solid #2867a8;
    box-shadow: 0 0 24px #0878ff30;
}

.brand-name {
    font-size: 23px;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(90deg, var(--cyan), #9a9cff);
    color: transparent;
    background-clip: text;
}

.brand-sub {
    font-size: 11px;
    color: var(--muted);
}

.nav {
    display: grid;
    gap: 9px;
}

.nav-item {
    padding: 12px;
    color: #a6b6d5;
    border: 1px solid transparent;
    border-radius: 11px;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 14px;
}

.nav-item.active {
    color: var(--cyan);
    border-color: #1878d9;
    background: linear-gradient(100deg, #0c3560, #15173b);
    box-shadow: inset 3px 0 var(--cyan), 0 0 20px #007bff14;
}

.nav-symbol { width: 22px; text-align: center; font-size: 18px; }

.system-card {
    margin-top: auto;
    padding: 13px;
    background: #091529;
    border: 1px solid var(--line);
    border-radius: 12px;
    color: var(--muted);
    font-size: 12px;
}

.online-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 7px;
    border-radius: 50%;
    background: #36f5bb;
    box-shadow: 0 0 10px #36f5bb;
}

.main { min-width: 0; padding: 24px; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 22px;
}

.eyebrow {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 10px;
    font-weight: 600;
}

h1 { font-size: 25px; margin: 5px 0; }
.subtitle { color: var(--muted); font-size: 13px; }

.status-pill {
    border: 1px solid #206b9b;
    background: #0b2038;
    color: var(--cyan);
    border-radius: 30px;
    padding: 8px 12px;
    font-size: 12px;
    white-space: nowrap;
}

.dashboard {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 250px;
    gap: 18px;
    align-items: start;
}

.primary { min-width: 0; display: grid; gap: 18px; }

.panel {
    min-width: 0;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: linear-gradient(145deg, #0c1730ee, #080d1dee);
    box-shadow: 0 12px 38px #00000025;
    overflow: hidden;
}

.panel-heading {
    padding: 17px 19px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}

.panel-title { font-weight: 600; font-size: 15px; }
.panel-desc { color: var(--muted); font-size: 11px; margin-top: 4px; }

.section-icon {
    color: var(--cyan);
    font-size: 21px;
    margin-right: 9px;
}

.panel-body { padding: 0 19px 18px; }

.settings-grid {
    display: grid;
    grid-template-columns: 1fr 1.35fr;
    gap: 14px;
}

label {
    display: block;
    font-size: 12px;
    color: #a7bddf;
    margin: 0 0 8px;
}

input, textarea {
    width: 100%;
    border: 1px solid #203b68;
    color: var(--text);
    background: #050b1b;
    border-radius: 10px;
    padding: 12px 13px;
    outline: none;
    transition: border .2s, box-shadow .2s;
}

input:focus, textarea:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px #38e8ff12;
}

input { min-height: 44px; }
.settings-grid textarea { min-height: 70px; resize: vertical; }

.primary-button {
    border: 1px solid #629dff;
    border-radius: 10px;
    padding: 11px 16px;
    color: white;
    font-weight: 600;
    background: linear-gradient(105deg, #1267d8, #6739e8);
    box-shadow: 0 0 22px #425eff27;
    transition: transform .2s, filter .2s;
}

.primary-button:hover { filter: brightness(1.18); transform: translateY(-1px); }

.secondary-button {
    background: #0a1730;
    color: #a9c9f7;
    border: 1px solid #25456e;
    border-radius: 9px;
    padding: 9px 12px;
}

.settings-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 12px;
}

.chat-panel { display: flex; flex-direction: column; min-height: 600px; }

.chat-heading { border-bottom: 1px solid #162b4b; }

.chat-heading-left { display: flex; align-items: center; gap: 10px; }

.ai-avatar {
    width: 37px;
    height: 37px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, #0d5a85, #512ca5);
    color: white;
    border: 1px solid #3685db;
    box-shadow: 0 0 18px #1768ff24;
    flex-shrink: 0;
}

.chat-label { color: #78a7ff; font-size: 10px; margin-top: 3px; }

.chat {
    padding: 20px;
    flex: 1;
    min-height: 350px;
    max-height: 520px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
    background-image: radial-gradient(#1d3c6a 0.7px, transparent 0.7px);
    background-size: 22px 22px;
}

.message-row { display: flex; align-items: flex-start; gap: 10px; }
.message-row.user-row { justify-content: flex-end; }

.message {
    max-width: 85%;
    padding: 13px 15px;
    border-radius: 4px 15px 15px 15px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.8;
    font-size: 13px;
    background: #101d37;
    border: 1px solid #203a65;
    color: #d8e6ff;
}

.message {
    white-space: normal;
}

.message-content p {
    margin: 0 0 10px;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-content h1,
.message-content h2,
.message-content h3,
.message-content h4 {
    line-height: 1.5;
    margin: 12px 0 7px;
    color: #eaf3ff;
}

.message-content ul,
.message-content ol {
    padding-left: 24px;
    margin: 8px 0;
}

.message-content li {
    margin: 4px 0;
}

.message-content blockquote {
    margin: 10px 0;
    padding: 8px 12px;
    border-left: 3px solid var(--cyan);
    background: #0a1730;
    color: #b9cbea;
}

.message-content code {
    overflow-wrap: anywhere;
    padding: 2px 5px;
    border-radius: 5px;
    background: #050b1b;
    color: #9feaff;
}

.message-content pre {
    max-width: 100%;
    overflow-x: auto;
    padding: 12px;
    border: 1px solid #203a65;
    border-radius: 9px;
    background: #050b1b;
}

.message-content pre code {
    padding: 0;
    background: transparent;
}

.message-content mjx-container {
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
}

.message-content mjx-container[display="true"] {
    margin: 0.8em 0;
}

.user-row .message {
    border-radius: 15px 4px 15px 15px;
    background: linear-gradient(135deg, #193b84, #34227a);
    border-color: #5261d5;
    color: white;
}

.message-meta { font-size: 10px; color: #7993bd; margin-top: 7px; }

.chat-composer {
    padding: 15px;
    border-top: 1px solid #1b3154;
    background: #070e20;
}

.composer-inner {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    border: 1px solid #28538a;
    border-radius: 13px;
    padding: 7px;
    background: #050a18;
    transition: border .2s, box-shadow .2s;
}

.composer-inner:focus-within {
    border-color: var(--cyan);
    box-shadow: 0 0 22px #00d9ff12;
}

.composer-inner textarea {
    flex: 1;
    min-width: 0;
    border: 0;
    box-shadow: none;
    background: transparent;
    resize: vertical;
    min-height: 43px;
    max-height: 140px;
}

.composer-inner textarea:focus { box-shadow: none; }

.send-button {
    width: 45px;
    height: 43px;
    flex-shrink: 0;
    border: 0;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    background: linear-gradient(145deg, #258eff, #773eff);
    box-shadow: 0 0 16px #485cff40;
}

.send-button:disabled { opacity: .5; cursor: wait; }

.status { color: var(--muted); font-size: 11px; min-height: 19px; padding: 6px 4px 0; }

.right-column { display: grid; gap: 16px; }

.subject-grid {
    padding: 0 13px 15px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px;
}

.subject {
    border: 1px solid #1c3a65;
    background: linear-gradient(145deg, #0c1c39, #090f23);
    border-radius: 11px;
    padding: 14px 5px;
    text-align: center;
    color: #c4d6f6;
    font-size: 11px;
    transition: .2s;
}

.subject:hover {
    border-color: var(--cyan);
    background: #10294a;
    transform: translateY(-2px);
}

.subject-symbol {
    font-size: 22px;
    display: block;
    color: var(--cyan);
    margin-bottom: 7px;
}

.notes-empty {
    margin: 0 14px 15px;
    padding: 15px;
    color: var(--muted);
    border: 1px dashed #294267;
    border-radius: 10px;
    font-size: 12px;
    line-height: 1.8;
}

.tip {
    position: relative;
    padding: 18px;
    border: 1px solid #5142ba;
    border-radius: 15px;
    overflow: hidden;
    background:
        radial-gradient(circle at 100% 0%, #592bd544, transparent 50%),
        linear-gradient(135deg, #111b45, #160f32);
}

.tip strong { display: block; margin-bottom: 7px; }
.tip p { color: #9faed3; font-size: 12px; line-height: 1.8; margin: 0; }

@media (min-width: 1400px) {
    .main { padding: 30px 34px; }
    .dashboard { grid-template-columns: minmax(0, 1fr) 275px; }
}

@media (max-width: 1050px) {
    .dashboard { grid-template-columns: minmax(0, 1fr); }
    .right-column { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .tip { grid-column: 1 / -1; }
}

@media (max-width: 700px) {
    .app { grid-template-columns: 1fr; }
    .sidebar {
        padding: 13px 15px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
        gap: 12px;
    }
    .brand { padding: 0; }
    .nav {
        display: flex;
        overflow-x: auto;
        gap: 7px;
    }
    .nav-item { white-space: nowrap; padding: 9px 11px; }
    .system-card { display: none; }
    .main { padding: 16px 11px; }
    .topbar { align-items: flex-start; }
    h1 { font-size: 21px; }
    .status-pill { font-size: 10px; }
    .settings-grid { grid-template-columns: 1fr; }
    .right-column { grid-template-columns: 1fr; }
    .tip { grid-column: auto; }
    .chat-panel { min-height: 560px; }
    .chat { padding: 12px; }
    .message { max-width: 92%; }
    .panel-heading { padding: 14px; }
    .panel-body { padding: 0 14px 15px; }
    .hide-mobile { display: none; }
}
</style>

<script>
window.MathJax = {
    tex: {
        inlineMath: [['\\(', '\\)'], ['$', '$']],
        displayMath: [['\\[', '\\]'], ['$$', '$$']],
        processEscapes: true
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    startup: { typeset: false }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

</head>

<body>
<div class="app">

    <aside class="sidebar">
        <div class="brand">
            <div class="brand-icon">⌘</div>
            <div>
                <div class="brand-name">NEURA</div>
                <div class="brand-sub">AI LEARNING SYSTEM</div>
            </div>
        </div>

        <nav class="nav">
            <div class="nav-item active"><span class="nav-symbol">⌂</span> ห้องเรียน</div>
            <div class="nav-item"><span class="nav-symbol">◷</span> ประวัติแชต</div>
            <div class="nav-item"><span class="nav-symbol">▤</span> โน้ตการเรียน</div>
            <div class="nav-item"><span class="nav-symbol">⚙</span> ตั้งค่าผู้สอน</div>
        </nav>

        <div class="system-card">
            <span class="online-dot"></span> SYSTEM OPERATIONAL
            <div style="margin-top:8px">NEURA INTERFACE · v1.0</div>
        </div>
    </aside>

    <main class="main">
        <header class="topbar">
            <div>
                <div class="eyebrow">PERSONALIZED INTELLIGENCE</div>
                <h1>ห้องเรียนแห่งอนาคต<span style="color:var(--cyan)">.</span></h1>
                <div class="subtitle">เรียนรู้ได้ทุกเรื่อง ในแบบที่เป็นคุณ</div>
            </div>
            <div class="status-pill">✦ GEMINI AI</div>
        </header>

        <div class="dashboard">
            <div class="primary">

                <section class="panel">
                    <div class="panel-heading">
                        <div>
                            <div class="panel-title"><span class="section-icon">◈</span>กำหนดตัวตนผู้สอน</div>
                            <div class="panel-desc">ปรับบุคลิกและรูปแบบการอธิบายให้ตรงกับสไตล์ของคุณ</div>
                        </div>
                    </div>

                    <div class="panel-body">
                        <form id="personaForm">
                            <div class="settings-grid">
                                <div>
                                    <label for="persona_name">ชื่อผู้สอน / ตัวละคร</label>
                                    <input id="persona_name" maxlength="80"
                                           value="{{ persona_name }}" required
                                           placeholder="เช่น ไอน์สไตน์">
                                </div>
                                <div>
                                    <label for="persona_style">สไตล์การสอน</label>
                                    <textarea id="persona_style" maxlength="300"
                                              required
                                              placeholder="เช่น อธิบายเป็นขั้นตอน ยกตัวอย่าง และมีอารมณ์ขัน">{{ persona_style }}</textarea>
                                </div>
                            </div>
                            <div class="settings-actions">
                                <button class="primary-button" type="submit">✦ บันทึกและเริ่มแชตใหม่</button>
                            </div>
                        </form>
                    </div>
                </section>

                <section class="panel chat-panel">
                    <div class="panel-heading chat-heading">
                        <div class="chat-heading-left">
                            <div class="ai-avatar">✳</div>
                            <div>
                                <div class="panel-title">NEURA / AI Tutor</div>
                                <div class="chat-label">● CONTEXT-AWARE LEARNING</div>
                            </div>
                        </div>
                        <button type="button" class="secondary-button" id="newChat">↻ แชตใหม่</button>
                    </div>

                    <div id="chat" class="chat" aria-live="polite">
                        <div class="message-row">
                            <div class="ai-avatar">✳</div>
                            <div class="message">
                                <div style="color:var(--cyan);font-size:11px;margin-bottom:7px">NEURA · AI TUTOR</div>
                                สวัสดีครับ! พร้อมเริ่มการเรียนรู้หรือยัง?

                                วันนี้อยากเรียนเรื่องอะไร ไม่ว่าจะเป็นคณิตศาสตร์ วิทยาศาสตร์ ภาษา หรือการเขียนโปรแกรม ผมจะช่วยอธิบายทีละขั้นตอน พร้อมตัวอย่างที่เข้าใจง่ายครับ
                                <div class="message-meta">AI TUTOR · READY</div>
                            </div>
                        </div>
                    </div>

                    <div class="chat-composer">
                        <form id="chatForm">
                            <div class="composer-inner">
                                <textarea id="question" required
                                          placeholder="ส่งคำถามเข้าสู่ระบบ..."
                                          aria-label="พิมพ์คำถาม"></textarea>
                                <button class="send-button" id="sendButton"
                                        type="submit" aria-label="ส่งข้อความ">➤</button>
                            </div>
                            <div id="status" class="status" aria-live="polite">AI อาจตอบผิดได้ ควรตรวจสอบข้อมูลสำคัญ</div>
                        </form>
                    </div>
                </section>
            </div>

            <aside class="right-column">
                <section class="panel">
                    <div class="panel-heading">
                        <div>
                            <div class="panel-title"><span class="section-icon">⌘</span>เลือกหมวดวิชา</div>
                            <div class="panel-desc">ระบบรองรับการเรียนรู้หลากหลายด้าน</div>
                        </div>
                    </div>

                    <div class="subject-grid">
                        <button class="subject" type="button" data-subject="คณิตศาสตร์"><span class="subject-symbol">∑</span>คณิตศาสตร์</button>
                        <button class="subject" type="button" data-subject="วิทยาศาสตร์"><span class="subject-symbol">⚛</span>วิทยาศาสตร์</button>
                        <button class="subject" type="button" data-subject="ภาษาอังกฤษ"><span class="subject-symbol">文</span>ภาษาอังกฤษ</button>
                        <button class="subject" type="button" data-subject="ภาษาไทย"><span class="subject-symbol">ก</span>ภาษาไทย</button>
                        <button class="subject" type="button" data-subject="สังคมศึกษา"><span class="subject-symbol">◎</span>สังคมศึกษา</button>
                        <button class="subject" type="button" data-subject="การเขียนโปรแกรม"><span class="subject-symbol">&lt;/&gt;</span>เขียนโปรแกรม</button>
                        <button class="subject" type="button" data-subject="ศิลปะ"><span class="subject-symbol">✳</span>ศิลปะ</button>
                        <button class="subject" type="button" data-subject="สุขศึกษา"><span class="subject-symbol">♡</span>สุขศึกษา</button>
                    </div>
                </section>

                <section class="panel">
                    <div class="panel-heading">
                        <div>
                            <div class="panel-title"><span class="section-icon">▤</span>พื้นที่การเรียนรู้</div>
                            <div class="panel-desc">บันทึกและทบทวนความรู้</div>
                        </div>
                    </div>
                    <div class="notes-empty">
                        <div style="font-size:25px;color:var(--cyan);margin-bottom:8px">⌁</div>
                        <strong style="color:#d7e7ff">Study Notes</strong>
                        <p>คุณสามารถสั่ง AI ให้บันทึกโน้ตหรือแสดงโน้ตที่บันทึกไว้ผ่านช่องแชตได้</p>
                    </div>
                </section>

                <section class="tip">
                    <div style="font-size:22px;margin-bottom:9px">✧</div>
                    <strong>LEARN WITHOUT LIMITS</strong>
                    <p>ทุกคำถามคือจุดเริ่มต้นของความเข้าใจใหม่ เรียนรู้ตามจังหวะของคุณเอง</p>
                </section>
            </aside>
        </div>
    </main>
</div>

<script>
const chat = document.getElementById("chat");
const question = document.getElementById("question");
const statusBox = document.getElementById("status");
const sendButton = document.getElementById("sendButton");

function addMessage(text, role) {
    const row = document.createElement("div");
    row.className = "message-row" + (role === "user" ? " user-row" : "");

    if (role !== "user") {
        const avatar = document.createElement("div");
        avatar.className = "ai-avatar";
        avatar.textContent = "✳";
        row.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = "message";

    // ข้อความผู้ใช้แสดงเป็นข้อความธรรมดา ส่วนคำตอบ AI แปลง Markdown/LaTeX
    if (role === "user") {
        bubble.textContent = text;
    } else {
        const content = document.createElement("div");
        content.className = "message-content";

        if (window.marked && window.DOMPurify) {
            const markdownHTML = window.marked.parse(text, {
                gfm: true,
                breaks: true
            });
            content.innerHTML = window.DOMPurify.sanitize(markdownHTML);
        } else {
            // หาก CDN โหลดไม่สำเร็จ ยังแสดงคำตอบได้อย่างปลอดภัย
            content.textContent = text;
        }

        bubble.appendChild(content);

        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([content]).catch((error) => {
                console.error("MathJax rendering error:", error);
            });
        }
    }

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = role === "user" ? "YOU" : "NEURA · AI TUTOR";
    bubble.appendChild(document.createElement("br"));
    bubble.appendChild(meta);

    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

async function postJSON(url, data = {}) {
    const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.error || "เกิดข้อผิดพลาด");
    }

    return result;
}

document.getElementById("personaForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    statusBox.textContent = "กำลังบันทึกการตั้งค่า...";

    try {
        await postJSON("/persona", {
            persona_name: document.getElementById("persona_name").value,
            persona_style: document.getElementById("persona_style").value
        });
        window.location.reload();
    } catch (error) {
        statusBox.textContent = error.message;
    }
});

document.getElementById("newChat").addEventListener("click", async () => {
    try {
        await postJSON("/new-chat");
        window.location.reload();
    } catch (error) {
        statusBox.textContent = error.message;
    }
});

document.getElementById("chatForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = question.value.trim();
    if (!text || sendButton.disabled) return;

    addMessage(text, "user");
    question.value = "";
    sendButton.disabled = true;
    statusBox.textContent = "✦ NEURA กำลังประมวลผล...";

    try {
        const result = await postJSON("/chat", {message: text});
        addMessage(result.answer, "assistant");
        statusBox.textContent = "● พร้อมเรียนรู้ต่อ";
    } catch (error) {
        statusBox.textContent = error.message;
        addMessage("ไม่สามารถรับคำตอบได้ในขณะนี้ กรุณาตรวจสอบการเชื่อมต่อแล้วลองอีกครั้ง", "assistant");
    } finally {
        sendButton.disabled = false;
        question.focus();
    }
});

document.querySelectorAll("[data-subject]").forEach((button) => {
    button.addEventListener("click", () => {
        question.value = "ช่วยสอนเรื่อง" + button.dataset.subject +
            " แบบเข้าใจง่าย พร้อมยกตัวอย่างและแบบฝึกหัด";
        question.focus();
    });
});

question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        document.getElementById("chatForm").requestSubmit();
    }
});
</script>
</body>
</html>
"""



# -------------------------
# 7. Routes ของเว็บไซต์
# -------------------------

@app.route("/")
def index():
    conversation = get_current_conversation()

    return render_template_string(
        PAGE,
        persona_name=conversation["persona_name"],
        persona_style=conversation["persona_style"],
    )


@app.route("/persona", methods=["POST"])
def set_persona():
    data = request.get_json(silent=True) or {}

    name = str(data.get("persona_name", "")).strip()[:80]
    style = str(data.get("persona_style", "")).strip()[:300]

    if not name or not style:
        return jsonify({
            "error": "กรุณากรอกชื่อผู้สอนและสไตล์การสอน"
        }), 400

    session["persona_name"] = name
    session["persona_style"] = style
    session["conversation_id"] = create_conversation(name, style)

    return jsonify({"ok": True})


@app.route("/new-chat", methods=["POST"])
def new_chat():
    conversation = get_current_conversation()

    session["conversation_id"] = create_conversation(
        conversation["persona_name"],
        conversation["persona_style"],
    )

    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "กรุณาพิมพ์คำถาม"}), 400

    if len(message) > 5000:
        return jsonify({
            "error": "ข้อความยาวเกินไป กรุณาไม่เกิน 5,000 ตัวอักษร"
        }), 400

    conversation = get_current_conversation()

    try:
        answer = ask_ai(message, conversation)
        return jsonify({"answer": answer})

    except Exception as error:
        app.logger.exception("AI request failed")

        # แสดงข้อความที่เข้าใจง่าย โดยไม่เปิดเผยรายละเอียดภายใน
        if not API_KEY:
            message = "ยังไม่ได้ตั้งค่า GEMINI_API_KEY ในไฟล์ .env"
        else:
            message = (
                "เรียกใช้งาน AI ไม่สำเร็จ กรุณาตรวจสอบ API Key "
                "การเชื่อมต่ออินเทอร์เน็ต และชื่อโมเดล "
                "แล้วลองใหม่อีกครั้ง"
            )

        return jsonify({"error": message}), 500


# -------------------------
# 8. เริ่มโปรแกรม
# -------------------------

init_db()

if __name__ == "__main__":
    app.run(debug=True)

"""น้องกู้ — แชตบอต กยศ. ศิลปากร (RAG) สำหรับ Hugging Face Spaces
ย้ายมาจากโน้ตบุ๊ก new_chatbot_rag_colab.ipynb (ตัดส่วนข้อสอบ/Chroma/โค้ดทดลองออก เหลือเฉพาะส่วนที่ใช้รันจริง)
ต้องตั้ง Secret ชื่อ GEMINI_API_KEY ในหน้า Settings ของ Space
"""
import os, re, json, time
from pathlib import Path

import numpy as np
import gradio as gr
import spaces

@spaces.GPU
def _gpu_ping():
    return True
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

# ---------- ค่าตั้ง ----------
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")   # เปลี่ยนได้ผ่าน Variable ชื่อ GEMINI_MODEL
EMBED_MODEL_NAME = "BAAI/bge-m3"
DATA_PATH = Path(__file__).parent / "data" / "cards.jsonl"
MAX_INPUT_CHARS = 1000      # กันข้อความยาวเกิน (ประหยัดโควตา)
MAX_HISTORY_MSGS = 12       # ส่งประวัติแชทล่าสุดให้โมเดลไม่เกินกี่ข้อความ

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------- เรียก Gemini ----------
def to_gemini(messages, temperature):
    """แปลงซองเป็นรูปแบบของ Gemini: กฎประจำตัว (system) แยกออกมา ที่เหลือคือบทสนทนา"""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    return contents, {"system_instruction": system, "temperature": temperature}


def call_llm(messages, temperature=0.3):
    """ส่งคำถามให้ Gemini และลองใหม่เมื่อเจอ 429 หรือ 503"""

    if client is None:
        raise RuntimeError("ยังไม่ได้ตั้ง Secret GEMINI_API_KEY")

    contents, config = to_gemini(messages, temperature)

    for wait in [5, 10, 20, 0]:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config
            )

            return response.text or ""

        except errors.APIError as e:

            if e.code not in (429, 503) or wait == 0:
                raise

            print(f"Gemini error {e.code}: รอ {wait} วินาทีแล้วลองใหม่")

            time.sleep(wait)


# ---------- โหลดการ์ด + สร้างแผนที่ ----------
AUD_OLD, AUD_NEW, AUD_ALL = "ผู้กู้รายเก่า", "ผู้กู้รายใหม่", "ทุกกลุ่ม"

all_cards = []
for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    m = r["metadata"]
    all_cards.append({"id": r["id"], "section": m["section"], "text": r["text"],
                      "audience": m.get("audience", AUD_ALL), "topic": m.get("topic", ""),
                      "page": m.get("page", "-"), "term": m.get("term", "2569/1")})
print(f"โหลดการ์ด {len(all_cards)} ใบ")

print("โหลดเครื่องแปลความหมาย", EMBED_MODEL_NAME, "...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


def embed(texts):
    """ข้อความ → พิกัด (normalize แล้ว ทำให้ dot product = cosine similarity)"""
    return embedder.encode(list(texts), normalize_embeddings=True, batch_size=16)


card_vecs = embed([c["section"] + "\n" + c["text"] for c in all_cards])
print("แผนที่:", card_vecs.shape)


def search(query, k=4, audience=None):
    allowed = {audience, AUD_ALL} if audience else None
    q = embed([query])[0]
    scores = card_vecs @ q
    results = []
    for i in np.argsort(-scores):
        c = all_cards[i]
        if allowed and c["audience"] not in allowed:
            continue
        results.append({**c, "score": round(float(scores[i]), 3)})
        if len(results) == k:
            break
    return results


# ---------- RAG ----------
SYSTEM_RULES = """คุณคือ "น้องกู้" ผู้ช่วยตอบคำถามเรื่องการกู้ยืม กยศ. ของนักศึกษามหาวิทยาลัยศิลปากร วิทยาเขตพระราชวังสนามจันทร์
- ตอบเป็นภาษาไทย สุภาพ กระชับ อ่านง่าย
- ถ้าไม่แน่ใจ ให้บอกตรง ๆ ว่าไม่ทราบ
- ตอบตรงคำถามเลย ไม่ต้องทักทายหรือแนะนำตัวทุกครั้ง"""

def build_envelope(question, system_rules=SYSTEM_RULES, tools="", memory="", cards="", history=None):
    """กฎประจำตัว → เครื่องมือ → สมุดบันทึก → การ์ดจากประกาศ → ประวัติแชท → คำถาม"""
    system = system_rules
    if tools:
        system += "\n\n[เครื่องมือที่ใช้ได้]\n" + tools
    if memory:
        system += "\n\n[บันทึกความจำ]\n" + memory
    if cards:
        system += "\n\n[การ์ดจากประกาศ กยศ.]\n" + cards
    return [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": question}]

RAG_RULES = SYSTEM_RULES + """

กติกาบนผนัง:
1) ตอบจาก [การ์ดจากประกาศ กยศ.] เท่านั้น และบอกว่าใช้การ์ดหัวข้อไหน
2) ถ้าในการ์ดไม่มีคำตอบ ให้ตอบว่า "ประกาศที่ผมมีไม่ได้เขียนเรื่องนี้ไว้ครับ ลองสอบถามกองกิจการนักศึกษา ห้อง Su Service นะครับ"
3) ห้ามเดาวันที่ จำนวนเงิน หรือขั้นตอนที่ไม่มีในการ์ด ถ้าการ์ดเขียนว่า "จะแจ้งให้ทราบอีกครั้ง" ให้บอกตามนั้น
4) การ์ดมีสองกลุ่ม คือ ผู้กู้รายเก่า และ ผู้กู้รายใหม่ ซึ่งเงื่อนไขต่างกัน ถ้าคำถามขึ้นกับกลุ่มและการ์ดมีทั้งสองกลุ่ม ให้ตอบแยกเป็นสองกลุ่ม หรือถามกลับว่าเป็นกลุ่มไหน
5) ถ้าข้อมูลยังไม่พอจะสรุป ให้บอกว่ารู้อะไรแล้ว และยังขาดข้อมูลอะไร
6) ถ้าคำถามตั้งสมมติฐานผิด ให้แก้ให้ถูกอย่างสุภาพ
7) ข้อมูลทั้งหมดเป็นของปีการศึกษา 2569 ภาคการศึกษาต้น ให้บอกข้อนี้ทุกครั้งที่ตอบเรื่องวันที่
8) ถ้าเป็นการทักทายหรือเล่าเรื่องตัวเอง ไม่ใช่คำถามเรื่อง กยศ. ให้ตอบรับสั้น ๆ และจำสิ่งที่เขาบอกจากประวัติแชท"""

def format_cards(results):
    return "\n\n".join(f"[{r['section']} | {r['audience']} | ปีการศึกษา {r['term']}]\n{r['text']}" for r in results)

def rag_answer(question, audience=None, k=4, history=None):
    results = search(question, k=k, audience=audience)
    msgs = build_envelope(question, system_rules=RAG_RULES, cards=format_cards(results), history=history)
    return call_llm(msgs, temperature=0), results


# ---------- ขั้นตอน / ตรวจกลุ่มผู้กู้ ----------
CARD_BY_ID = {c["id"]: c for c in all_cards}

# การ์ดที่เป็น "ขั้นตอน" ของแต่ละกลุ่ม เรียงตามลำดับที่ควรเล่า
PROCESS_IDS = {
    AUD_OLD: ["kysr69-021-timeline-summary",
              "kysr69-002-step1-submit-docs", "kysr69-003-step1-check-list", "kysr69-004-step1-over-years",
              "kysr69-005-step2-record-amount", "kysr69-006-step2-submit-withdraw-form",
              "kysr69-007-add-drop-amount", "kysr69-008-add-drop-submit",
              "kysr69-009-over-years-withdraw", "kysr69-010-after-approval",
              "kysr69-011-docs-application-form", "kysr69-012-docs-income", "kysr69-013-docs-volunteer",
              "kysr69-014-docs-attach-withdraw-form", "kysr69-015-consequence-no-submit",
              "kysr69-022-common-mistakes"],
    AUD_NEW: ["kysr69new-006-timeline-summary",
              "kysr69new-002-step1-preapprove", "kysr69new-007-docs-borrower", "kysr69new-008-docs-family-income",
              "kysr69new-009-income-certifier-rule", "kysr69new-010-docs-credit-bureau-parents",
              "kysr69new-011-copy-signature-rule", "kysr69new-003-step2-announce", "kysr69new-004-step3-dsl",
              "kysr69new-016-dsl-overview", "kysr69new-017-dsl-borrower-consent", "kysr69new-018-dsl-parent-consent",
              "kysr69new-019-dsl-income-docs", "kysr69new-020-dsl-welfare-card", "kysr69new-021-dsl-ksk-scholarship",
              "kysr69new-005-step4-contract", "kysr69new-022-dsl-contract", "kysr69new-023-dsl-withdraw-confirm"],
}
for aud, ids in PROCESS_IDS.items():
    missing = [i for i in ids if i not in CARD_BY_ID]
    assert not missing, f"ไม่พบการ์ด {missing}"

GREETING = """สวัสดีครับ ผมน้องกู้ ผู้ช่วยตอบเรื่องกู้ยืม กยศ. ของ ม.ศิลปากร วิทยาเขตพระราชวังสนามจันทร์ (ข้อมูลจากประกาศปีการศึกษา 2569 ภาคการศึกษาต้น)

ก่อนเริ่ม ขอถามก่อนนะครับ คุณเป็นผู้กู้กลุ่มไหน
- **ผู้กู้รายเก่า** = ทำสัญญากู้กับ ม.ศิลปากรไว้ก่อนปีการศึกษา 2569 และไม่ได้ย้ายสถานศึกษา
- **ผู้กู้รายใหม่** = ยังไม่เคยกู้ หรือเคยกู้จากสถานศึกษาอื่นและยังไม่ได้ทำสัญญากับ ม.ศิลปากร

พิมพ์ "รายเก่า" หรือ "รายใหม่" ได้เลยครับ"""

ASK_TEXT = """ขอทราบก่อนนะครับ คุณเป็นผู้กู้ **รายเก่า** หรือ **รายใหม่**
(รายเก่า = ทำสัญญากับ ม.ศิลปากรไว้ก่อนปี 2569 / รายใหม่ = ยังไม่เคยกู้ หรือมาจากสถานศึกษาอื่น)
พิมพ์ "รายเก่า" หรือ "รายใหม่" ได้เลย แล้วผมจะตอบคำถามของคุณต่อทันที"""

GROUP_NEW = r"ยังไม่เคยกู้|ไม่เคยกู้|ครั้งแรก|รายใหม่|ผู้กู้ใหม่|กู้ใหม่"
GROUP_OLD = r"รายเก่า|ผู้กู้เก่า|กู้เก่า|(?<!ไม่)เคยกู้|เคยทำสัญญา"
OTHER_SCHOOL = r"สถานศึกษาอื่น|มหาวิทยาลัยอื่น|ที่อื่น|ย้ายมา"      # รายเก่าจากที่อื่น = นับเป็นรายใหม่
FILLER = r"ผม|ฉัน|หนู|พี่|น้อง|เป็น|คือ|ครับ|ค่ะ|คะ|นะ|ผู้กู้|กลุ่ม|ราย|เก่า|ใหม่|\s|[,.!?]"
STEP_PATTERN = r"ขั้นตอน|ต้องทำอะไร|ต้องทำยังไง|ต้องทำอย่างไร|ต้องทำไงบ้าง|ทำยังไงบ้าง|ทำอย่างไรบ้าง|ไทม์ไลน์|เริ่มยังไง|เริ่มอย่างไร|ขั้นไหน"
GREET_PATTERN = r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)"

def detect_audience(text):
    t = text.strip()
    if t in ("ใหม่", "เก่า"):
        return AUD_NEW if t == "ใหม่" else AUD_OLD
    if re.search(OTHER_SCHOOL, t):
        return AUD_NEW
    is_new, is_old = bool(re.search(GROUP_NEW, t)), bool(re.search(GROUP_OLD, t))
    if is_new and not is_old:
        return AUD_NEW
    if is_old and not is_new:
        return AUD_OLD
    return None                                          # ไม่พูดถึง หรือพูดถึงทั้งสองกลุ่ม (กำกวม)

def only_group(text):
    """ข้อความนี้บอกแค่กลุ่ม ไม่ได้ถามอะไรเพิ่ม เช่น 'รายเก่า' หรือ 'ผมเป็นผู้กู้รายใหม่ครับ'"""
    rest = re.sub(GROUP_NEW + "|" + GROUP_OLD + "|" + OTHER_SCHOOL, "", text)
    return len(re.sub(FILLER, "", rest)) == 0

def text_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(text_of(i) for i in x)
    if isinstance(x, dict):
        return str(x.get("text") or x.get("content") or "")
    return str(x)

def to_hist(history):
    """รองรับประวัติแชททั้งแบบ dict (Gradio รุ่นใหม่) และแบบคู่ [ผู้ใช้, บอท] (รุ่นเก่า)"""
    hist = []
    for h in history:
        if isinstance(h, dict):
            hist.append({"role": h["role"], "content": text_of(h["content"])})
        else:
            if h[0] is not None:
                hist.append({"role": "user", "content": text_of(h[0])})
            if h[1] is not None:
                hist.append({"role": "assistant", "content": text_of(h[1])})
    return hist

def trim_start(hist):
    """Gemini ต้องเริ่มบทสนทนาด้วยฝั่งผู้ใช้ จึงตัดข้อความทักทายของบอทที่นำหน้าออก"""
    i = 0
    while i < len(hist) and hist[i]["role"] != "user":
        i += 1
    return hist[i:]

STEPS_RULES = RAG_RULES + """

กติกาเพิ่มสำหรับคำถามขอ "ขั้นตอนทั้งหมด":
- การ์ดที่ให้ไปคือขั้นตอนทั้งหมดของกลุ่มที่ผู้ใช้ระบุ ให้สรุปเป็นลำดับขั้น เรียงตามเวลา ห้ามข้ามขั้นใด
- แต่ละขั้นให้บอก (1) ต้องทำอะไร (2) กำหนดเวลา (3) เอกสารที่ต้องใช้ (ถ้ามี)
- ขั้นไหนที่ประกาศบอกว่าจะแจ้งกำหนดการภายหลัง ให้ระบุตามนั้น ห้ามเดาวัน
- ใช้รายการเลขข้อ ภาษาไทย กระชับ ปิดท้ายด้วยข้อควรระวังและผลถ้าไม่ทำตามกำหนด (ถ้ามีในการ์ด)
- เริ่มด้วยประโยคเดียวว่านี่คือขั้นตอนของกลุ่มไหน ปีการศึกษา 2569 ภาคการศึกษาต้น"""

def steps_answer(audience, question, history=None):
    """ส่งการ์ดขั้นตอนของกลุ่มนั้น 'ทุกใบ' ให้บอท เพื่อให้ตอบครบทุกขั้น"""
    cards = [CARD_BY_ID[i] for i in PROCESS_IDS[audience]]
    msgs = build_envelope(question, system_rules=STEPS_RULES, cards=format_cards(cards), history=history)
    return call_llm(msgs, temperature=0)


def _chat_fn(message, history):
    message = text_of(message)
    hist = to_hist(history)
    aud = None
    for t in [h["content"] for h in hist if h["role"] == "user"] + [message]:
        aud = detect_audience(t) or aud                  # จำกลุ่มจากประวัติ ข้อความล่าสุดชนะ
    if aud is None:
        return ASK_TEXT                                  # ยังไม่รู้กลุ่ม ถามก่อน

    question, llm_hist = message, trim_start(hist)
    if only_group(message):                              # ผู้ใช้แค่บอกกลุ่ม
        pending = None
        if (len(hist) >= 2 and hist[-1]["role"] == "assistant" and hist[-1]["content"] == ASK_TEXT
                and hist[-2]["role"] == "user"):
            cand = hist[-2]["content"]
            if len(cand.strip()) > 3 and not re.search(GREET_PATTERN, cand.strip(), re.I):
                pending = cand                           # มีคำถามที่ค้างอยู่ ให้ตอบคำถามนั้นเลย
        if pending is None:
            return (f"รับทราบครับ คุณเป็น**{aud}** ถามได้เลยครับ เช่น\n"
                    "- ต้องทำขั้นตอนอะไรบ้าง\n- ต้องใช้เอกสารอะไรบ้าง\n- ส่งเอกสารได้ถึงวันไหน\n- คุณสมบัติผู้กู้เป็นอย่างไร")
        question, llm_hist = pending, trim_start(hist[:-2])

    if re.search(STEP_PATTERN, question):
        return steps_answer(aud, question, llm_hist)      # ถามขั้นตอน → ตอบครบทั้งกลุ่ม
    return rag_answer(question, audience=aud, history=llm_hist)[0]


def chat_fn(message, history):
    """ตัวห่อ: จำกัดความยาว/ประวัติ และจับ error ให้ผู้ใช้เห็นข้อความที่อ่านรู้เรื่อง"""
    message = text_of(message).strip()[:MAX_INPUT_CHARS]
    if not message:
        return "พิมพ์คำถามได้เลยครับ"
    history = (history or [])[-MAX_HISTORY_MSGS:]
    try:
        return _chat_fn(message, history)
    except errors.APIError as e:
        print("Gemini error:", e.code, e)
        if e.code == 429:
            return "ตอนนี้มีคนใช้งานเยอะ โควตาเต็มชั่วคราว ลองใหม่อีกสักครู่นะครับ"
        return "ขออภัยครับ ระบบตอบกลับขัดข้องชั่วคราว ลองใหม่อีกครั้งนะครับ"
    except Exception as e:
        print("Unexpected error:", repr(e))
        return "ขออภัยครับ เกิดข้อผิดพลาดบางอย่าง ลองใหม่อีกครั้งนะครับ"


# ---------- หน้าเว็บ ----------
def make_chatbot():
    seed = [{"role": "assistant", "content": GREETING}]
    for kw in ({"type": "messages"}, {}):          # รองรับ Gradio หลายรุ่น
        try:
            return gr.Chatbot(value=seed, height=520, **kw)
        except Exception:
            continue
    return None


def make_demo():
    base = dict(title="น้องกู้ ผู้ช่วย กยศ. ศิลปากร (ข้อมูลปีการศึกษา 2569 ภาคต้น)",
                description="ถามเรื่องกู้ยืม กยศ. ได้เลย น้องกู้จะถามก่อนว่าคุณเป็นผู้กู้รายเก่าหรือรายใหม่ "
                            "ข้อมูลจากประกาศกองกิจการนักศึกษา ควรตรวจสอบกับกองกิจการนักศึกษาอีกครั้งก่อนส่งเอกสารจริง")
    bot = make_chatbot()
    attempts = [dict(chatbot=bot, type="messages"), dict(chatbot=bot), {}] if bot is not None else [{}]
    for extra in attempts:
        try:
            return gr.ChatInterface(chat_fn, **base, **extra)
        except Exception:
            continue
    raise RuntimeError("สร้างหน้าแชตไม่สำเร็จ")


demo = make_demo()

if __name__ == "__main__":
    demo.launch()

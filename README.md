---
title: น้องกู้ กยศ. ศิลปากร
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.38.0
python_version: "3.11"
app_file: app.py
startup_duration_timeout: 1h
pinned: false
---

# น้องกู้ — แชตบอตตอบเรื่อง กยศ. ม.ศิลปากร (RAG)

ตอบคำถามจากประกาศ กยศ. ปีการศึกษา 2569 ภาคต้น (ผู้กู้รายเก่า / รายใหม่)
ค้นการ์ดด้วย `BAAI/bge-m3` แล้วให้ Gemini ตอบจากการ์ดเท่านั้น

## ตั้งค่าบน Hugging Face
1. Settings → **Variables and secrets** → New secret ชื่อ `GEMINI_API_KEY`
2. (ไม่บังคับ) New variable ชื่อ `GEMINI_MODEL` ถ้าต้องการเปลี่ยนรุ่น (ค่าเริ่มต้น `gemini-3.5-flash-lite`)

## อัปเดตข้อมูล
แก้ไฟล์ `data/cards.jsonl` (1 บรรทัด = 1 การ์ด) แล้ว commit Space จะ restart และ embed ใหม่เอง

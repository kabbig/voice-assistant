# main.py
import os
import time
import json
import tempfile
import uvicorn
import difflib
from collections import defaultdict
from typing import Optional
import requests
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import openai

# Загружаем .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
VOX_SIGNATURE_SECRET = os.getenv("VOX_SIGNATURE_SECRET", "")

openai.api_key = OPENAI_API_KEY

app = FastAPI()



# ===== FAQ и промпт =====
def load_faq():
    try:
        with open("faq.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def load_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

FAQ = load_faq()
PROMPT = load_prompt()

# ===== Память контекста =====
SESSIONS = defaultdict(list)
LAST_ACTIVITY = {}
SESSION_TTL = 300  # 5 минут

def update_session(caller: str, role: str, content: str):
    SESSIONS[caller].append({"role": role, "content": content})
    LAST_ACTIVITY[caller] = time.time()

def clear_session(caller: str):
    SESSIONS.pop(caller, None)
    LAST_ACTIVITY.pop(caller, None)

# ====== FAQ поиск ======
def find_faq_answer(user_text: str, cutoff=0.6) -> Optional[str]:
    user_norm = user_text.lower()
    questions = [q["question"].lower() for q in FAQ]
    matches = difflib.get_close_matches(user_norm, questions, n=1, cutoff=cutoff)
    if matches:
        for q in FAQ:
            if q["question"].lower() == matches[0]:
                return q["answer"]
    return None

# ====== STT (Speech-to-Text) ======
def yandex_stt(file_path: str) -> str:
    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    headers = {"Authorization": f"Bearer {YANDEX_IAM_TOKEN}"}
    with open(file_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f)
    try:
        return resp.json().get("result", "")
    except:
        return ""

# ====== TTS (Text-to-Speech) ======
def yandex_tts(text: str) -> str:
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Bearer {YANDEX_IAM_TOKEN}"}
    data = {
        "text": text,
        "lang": "ru-RU",
        "voice": "alena",
        "format": "mp3"
    }
    r = requests.post(url, headers=headers, data=data)
    filename = f"static/{int(time.time())}.mp3"
    os.makedirs("static", exist_ok=True)
    with open(filename, "wb") as f:
        f.write(r.content)
    return f"{BASE_URL}/{filename}"

# ====== GPT ответ ======
def ask_gpt(caller: str, user_text: str) -> str:
    update_session(caller, "user", user_text)
    messages = [{"role": "system", "content": PROMPT}] + SESSIONS[caller]
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400
    )
    answer = res.choices[0].message["content"]
    update_session(caller, "assistant", answer)
    return answer.strip()

# ====== Webhook от Voximplant ======
@app.post("/vox/webhook")
async def webhook(req: Request):
    data = await req.json()
    event = data.get("event")
    caller = data.get("callerid", "unknown")
    recording_url = data.get("recording_url")

    if event == "End":
        clear_session(caller)
        return {"status": "cleared"}

    if recording_url:
        # скачиваем запись
        r = requests.get(recording_url)
        tmp = tempfile.mktemp(suffix=".mp3")
        with open(tmp, "wb") as f:
            f.write(r.content)

        # распознаём
        user_text = yandex_stt(tmp)
        os.remove(tmp)

        if not user_text:
            reply_text = "Извините, я не расслышал. Повторите, пожалуйста."
        else:
            faq_answer = find_faq_answer(user_text)
            if faq_answer:
                reply_text = faq_answer
            else:
                reply_text = ask_gpt(caller, user_text)

        # синтезируем
        playback_url = yandex_tts(reply_text)
        return {"playback_url": playback_url, "text": reply_text}

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


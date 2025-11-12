import os
import time
import json
import tempfile
import asyncio
import aiohttp
import aioredis
from typing import Optional
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import openai
import uvicorn

load_dotenv()

# === ENV ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YANDEX_IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

YCLIENTS_PARTNER_TOKEN = os.getenv("YCLIENTS_PARTNER_TOKEN")
YCLIENTS_USER_LOGIN = os.getenv("YCLIENTS_USER_LOGIN")
YCLIENTS_USER_PASSWORD = os.getenv("YCLIENTS_USER_PASSWORD")
YCLIENTS_COMPANY_ID = os.getenv("YCLIENTS_COMPANY_ID")
YCLIENTS_SERVICE_IDS = [s for s in os.getenv("YCLIENTS_SERVICE_IDS", "").split(",") if s]
YCLIENTS_STAFF_IDS = [s for s in os.getenv("YCLIENTS_STAFF_IDS", "").split(",") if s]
TZ = os.getenv("TZ", "Europe/Moscow")
WORK_HOURS = os.getenv("WORK_HOURS", "09:00-18:00")
SLOT_DURATION_MIN = int(os.getenv("SLOT_DURATION_MIN", 30))
SLOT_BUFFER_MIN = int(os.getenv("SLOT_BUFFER_MIN", 10))

FAQ_PATH = os.getenv("FAQ_PATH", "faq.json")
PROMPT_PATH = os.getenv("PROMPT_PATH", "prompt.txt")

openai.api_key = OPENAI_API_KEY

app = FastAPI()
REDIS: Optional[aioredis.Redis] = None


# === REDIS ===
async def get_redis():
    global REDIS
    if not REDIS:
        REDIS = await aioredis.from_url("redis://localhost", decode_responses=True)
    return REDIS


# === FILE LOADERS ===
def load_faq():
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_prompt():
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


FAQ = load_faq()
PROMPT = load_prompt()


# === UTILS ===
async def retry_request(fn, retries=3, delay=1):
    for i in range(retries):
        try:
            return await fn()
        except Exception as e:
            if i < retries - 1:
                await asyncio.sleep(delay * (2 ** i))
            else:
                raise e


# === YCLIENTS ADAPTER ===
class YClientsAdapter:
    def __init__(self):
        self.partner_token = YCLIENTS_PARTNER_TOKEN
        self.company_id = YCLIENTS_COMPANY_ID
        self.user_token = None

    async def authorize(self):
        url = "https://api.yclients.com/api/v1/auth"
        data = {
            "login": YCLIENTS_USER_LOGIN,
            "password": YCLIENTS_USER_PASSWORD,
            "partner_token": self.partner_token
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=data) as r:
                js = await r.json()
                self.user_token = js.get("data", {}).get("user_token")

    async def _headers(self):
        if not self.user_token:
            await self.authorize()
        return {
            "Authorization": f"Bearer {self.user_token}, User {self.partner_token}",
            "Content-Type": "application/json"
        }

    async def get_slots(self, service_id: int, staff_id: int):
        async def _do():
            headers = await self._headers()
            async with aiohttp.ClientSession() as s:
                url = f"https://api.yclients.com/api/v1/book_dates/{self.company_id}/{staff_id}"
                async with s.get(url, headers=headers) as r:
                    days = await r.json()
                day = (days.get("data") or [])[0]
                if not day:
                    return []
                day_str = day["date"]
                url2 = f"https://api.yclients.com/api/v1/book_times/{self.company_id}/{staff_id}/{service_id}?date={day_str}"
                async with s.get(url2, headers=headers) as r:
                    times = await r.json()
                    slots = times.get("data", [])
                    return slots[:3]
        return await retry_request(_do)

    async def create_record(self, fullname: str, phone: str, service_id: int, staff_id: int, ts_unix: int, call_id: str):
        async def _do():
            headers = await self._headers()
            url = f"https://api.yclients.com/api/v1/records/{self.company_id}"
            data = {
                "phone": phone,
                "fullname": fullname,
                "services": [{"id": service_id}],
                "staff_id": staff_id,
                "datetime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts_unix)),
                "api_id": call_id
            }
            async with aiohttp.ClientSession() as s:
                async with s.post(url, headers=headers, json=data) as r:
                    return await r.json()
        return await retry_request(_do)


yclients = YClientsAdapter()


# === SPEECHKIT ===
async def yandex_stt(file_path: str) -> str:
    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    headers = {"Authorization": f"Bearer {YANDEX_IAM_TOKEN}"}
    async with aiohttp.ClientSession() as s:
        with open(file_path, "rb") as f:
            async with s.post(url, headers=headers, data=f) as r:
                js = await r.json()
                return js.get("result", "")


async def yandex_tts(text: str) -> str:
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Bearer {YANDEX_IAM_TOKEN}"}
    data = {"text": text, "lang": "ru-RU", "voice": "alena", "format": "mp3"}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, data=data) as r:
            content = await r.read()
            os.makedirs("static", exist_ok=True)
            filename = f"static/{int(time.time())}.mp3"
            with open(filename, "wb") as f:
                f.write(content)
            return f"{BASE_URL}/{filename}"


# === GPT LOGIC ===
async def ask_gpt(caller: str, text: str) -> dict:
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": text}
    ]
    res = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400
    )
    content = res.choices[0].message["content"].strip()
    # LLM возвращает JSON {"act": "...", "text": "...", "hints": [...]}
    try:
        data = json.loads(content)
    except Exception:
        data = {"act": "say", "text": content, "hints": []}
    return data


# === MAIN LOGIC ===
@app.post("/vox/webhook")
async def webhook(req: Request):
    data = await req.json()
    event = data.get("event")
    caller = data.get("callerid", "unknown")
    call_id = data.get("call_id")
    recording_url = data.get("recording_url")
    redis = await get_redis()

    if event == "End":
        await redis.delete(f"call:{call_id}:stage")
        await redis.delete(f"call:{call_id}:intent")
        await redis.delete(f"call:{call_id}:slots")
        return {"status": "cleared"}

    if event == "SpeechCaptured" and recording_url:
        tmp = tempfile.mktemp(suffix=".mp3")
        async with aiohttp.ClientSession() as s:
            async with s.get(recording_url) as r:
                content = await r.read()
                with open(tmp, "wb") as f:
                    f.write(content)

        user_text = await yandex_stt(tmp)
        os.remove(tmp)

        if not user_text:
            reply_text = "Извините, я не расслышал. Повторите, пожалуйста."
            playback_url = await yandex_tts(reply_text)
            return {"playback_url": playback_url, "text": reply_text}

        gpt_reply = await ask_gpt(caller, user_text)
        act = gpt_reply.get("act", "say")
        reply_text = gpt_reply.get("text", "")

        # === HANDLE ACTIONS ===
        if act == "show_slots":
            slots_all = []
            for sid in YCLIENTS_SERVICE_IDS:
                for stf in YCLIENTS_STAFF_IDS:
                    slots = await yclients.get_slots(int(sid), int(stf))
                    slots_all.extend(slots)
            await redis.set(f"call:{call_id}:slots", json.dumps(slots_all), ex=3600)
            reply_text = "Вот ближайшие доступные слоты: " + ", ".join(
                [s.get("time", "время") for s in slots_all[:3]]
            )

        elif act.startswith("book"):
            slots_raw = await redis.get(f"call:{call_id}:slots")
            if slots_raw:
                slots = json.loads(slots_raw)
                idx = int(act.replace("book", "").strip("()") or 0)
                choice = slots[idx] if idx < len(slots) else slots[0]
                result = await yclients.create_record(
                    fullname="Гость",
                    phone=caller,
                    service_id=YCLIENTS_SERVICE_IDS[0],
                    staff_id=YCLIENTS_STAFF_IDS[0],
                    ts_unix=int(time.time()) + 3600,
                    call_id=call_id
                )
                reply_text = "Запись успешно создана!" if result.get("success") else "Не удалось создать запись."

        elif act == "goodbye":
            reply_text = "Спасибо за звонок! До свидания."

        playback_url = await yandex_tts(reply_text)
        return {"playback_url": playback_url, "text": reply_text}

    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

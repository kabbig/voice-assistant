@echo off
echo Запуск голосового ассистента...
call venv\Scripts\activate
start cmd /k "npx localtunnel --port 8000"
python main.py
pause

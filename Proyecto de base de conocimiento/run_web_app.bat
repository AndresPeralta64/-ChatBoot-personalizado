@echo off
echo Installing requirements...
"C:\Users\andre\AppData\Local\Programs\Python\Python314\python.exe" -m pip install -r requirements.txt
echo Starting Web App...
start http://127.0.0.1:5000
"C:\Users\andre\AppData\Local\Programs\Python\Python314\python.exe" app.py
pause

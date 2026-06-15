@echo off
cd /d D:\mask-detection
D:\mask-detection\yolov8-env\Scripts\python.exe -m streamlit run D:\mask-detection\app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --server.fileWatcherType none
pause
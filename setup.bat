@echo off

IF NOT EXIST ".venv" (
    python -m venv .venv
)
CALL .venv\Scripts\activate
python -m pip install --upgrade pip



IF EXIST "requirements.txt" (
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found.
)
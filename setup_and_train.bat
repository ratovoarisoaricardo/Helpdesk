@echo off
echo =========================================
echo Installation des dependances...
echo =========================================
call venv\Scripts\pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Erreur lors de l'installation des dependances.
    pause
    exit /b %ERRORLEVEL%
)

echo =========================================
echo Lancement de l'entrainement du modele...
echo =========================================
call venv\Scripts\python train.py
if %ERRORLEVEL% neq 0 (
    echo Erreur lors de l'entrainement du modele.
    pause
    exit /b %ERRORLEVEL%
)

echo =========================================
echo Lancement de l'application Streamlit...
echo =========================================
call venv\Scripts\streamlit run app.py
pause

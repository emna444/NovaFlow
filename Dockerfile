
# 1. Image de base : Python
FROM python:3.12-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier les dépendances
COPY requirements.txt .

# 4. Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le code du projet
COPY . .

# 6. Port utilisé par Streamlit
EXPOSE 8501

# 7. Démarrer l'application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

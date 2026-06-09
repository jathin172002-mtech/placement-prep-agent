FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY week5/ ./week5/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "week5/app.py", "--server.address=0.0.0.0"]
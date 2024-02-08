FROM python:3.11.7

# WORKDIR /app
COPY requirements.txt ./

RUN pip install -r requirements.txt

ENTRYPOINT uvicorn --host 0.0.0.0 main:app --reload --port 1000

COPY . .
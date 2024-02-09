FROM python:3.11.7

WORKDIR /code

COPY . /code/
# 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
FROM python:3.12-alpine3.20

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN adduser -D app

USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

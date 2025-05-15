FROM python:3.12-alpine3.20

WORKDIR /fastapi_app

COPY . /fastapi_app

RUN pip install --no-cache-dir --upgrade -r /fastapi_app/requirements.txt

RUN adduser -D app

USER app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

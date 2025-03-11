FROM python:3.12-alpine3.20

WORKDIR /fastapi

COPY ./requirements.txt /fastapi/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /fastapi/requirements.txt

ENTRYPOINT ["fastapi", "run", "app/main.py", "--port", "80"]

FROM python:3.12-slim

WORKDIR /filter

COPY ./requirements.txt /filter/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "./filter.py"]
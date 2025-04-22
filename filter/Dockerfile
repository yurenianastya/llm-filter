FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /filter

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./filter
ENTRYPOINT ["python", "./filter.py"]
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /filter

RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends -y \
    python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m app

USER app

CMD ["python3", "main.py"]
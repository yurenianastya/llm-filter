FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /filter

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        python3 \
        python3-pip \
        python3-dev && \
    	rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "main.py"]
FROM python:3.11

RUN apt-get update && apt-get install -y \
    wget curl git build-essential\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1

RUN pip install -r requirements.txt

COPY . /app


CMD ["/bin/bash"]
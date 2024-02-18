FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y upgrade
RUN apt install -y python3-pip
RUN apt install -y wget

RUN mkdir -p /models/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf
RUN wget -P /models/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-13b-fast-instruct-q5_K_M.gguf

RUN apt-get install -y git-lfs && git lfs install
RUN git clone https://huggingface.co/intfloat/multilingual-e5-large /models/multilingual-e5-large

RUN pip install --upgrade pip
COPY requirements.txt .
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip install -r requirements.txt

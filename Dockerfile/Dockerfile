FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y upgrade
RUN apt install -y python3-pip

RUN pip install --upgrade pip
COPY requirements.txt .
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip install -r requirements.txt

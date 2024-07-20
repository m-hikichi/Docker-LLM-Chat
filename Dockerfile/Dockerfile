FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update
RUN apt install -y python3.10-dev python3-pip
RUN pip install --upgrade pip

WORKDIR /tmp
COPY requirements.txt .
RUN pip install -r requirements.txt

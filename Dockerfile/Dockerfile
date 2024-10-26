FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update
RUN apt install -y python3.12 python3-pip
# Enabling python3.12 to be used under the name 'python'
RUN ln -s /usr/bin/python3.12 /usr/bin/python
# Bypassing PEP 668 restrictions on package installation in the global environment
RUN echo "[global]\nbreak-system-packages = true" > /etc/pip.conf

WORKDIR /tmp
COPY requirements.txt .
RUN pip install -r requirements.txt

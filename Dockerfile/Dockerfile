FROM python:3.10

RUN mkdir -p /tmp/llama2-model
# Download the recommended model at https://github.com/ggerganov/llama.cpp/discussions/20941
RUN wget -P /tmp/llama2-model https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q5_K_M.bin

RUN pip install --upgrade pip
RUN pip install llama-cpp-python==0.1.77
RUN pip install gradio==4.13.0

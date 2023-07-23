FROM python:3.10

RUN mkdir -p /tmp/llama2-model
RUN wget -P /tmp/llama2-model https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin
RUN wget -P /tmp/llama2-model https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q8_0.bin

RUN pip install --upgrade pip
RUN pip install llama-cpp-python==0.1.74

#------------------------
#the image cannot be entirely build
#in the process of getting fixed

#build image --> docker build --tag=$IMAGE .
#------------------------

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY foodyai foodyai
COPY requirements.txt requirements.txt
COPY .env .env

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn foodyai.api.fast:app --host 0.0.0.0
#--port $PORT

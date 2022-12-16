FROM python:3.8.12-buster

COPY foodyai foodyai
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn foodyai.api.fast:app --host 0.0.0.0
#--port $PORT

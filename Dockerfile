FROM python:3.7-slim
RUN apt-get update
RUN pip install --upgrade pip
ADD app /app
WORKDIR /app
COPY models /app/models
COPY src /app/src
RUN pip install -r requirements.txt
ADD config.py /app/config.py
EXPOSE $PORT
CMD python service.py -p $PORT
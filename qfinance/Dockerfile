FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /usr/src/qfinance

RUN apt-get update
RUN apt-get -y install python3-tk

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
COPY ./datasets/IBM_unadjusted.csv ./stock_data.csv

ENV ENV "DOCKER"

CMD ["python3", "cli.py", "--data-file", "stock_data.csv"]

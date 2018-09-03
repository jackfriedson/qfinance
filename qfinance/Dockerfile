FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /usr/src/qfinance

RUN apt-get install python-support
RUN update-python-modules -a

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ./datasets/IBM_unadjusted.csv ./stock_data.csv

ENV ENV "DOCKER"

CMD ["python", "cli.py", "--data-file", "stock_data.csv"]

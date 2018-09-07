FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /usr/src/qfinance

RUN apt-get update && \
    apt-get -y install python3-tk

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ./datasets/IBM_unadjusted.csv ./stock_data.csv

ENV ENV "DOCKER"
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

CMD ["python", "cli.py", \
     "--data-file=stock_data.csv", \
     "--n-folds=10", \
     "--interval=H", \
     "--fee=0" \
  ]

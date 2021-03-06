FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /usr/src/qfinance

# Install python3-tk
RUN apt-get update \
    && apt-get -y install python3-tk

# Install TA-Lib
RUN apt-get -y install wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar xvfz ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ./datasets/portfolio_data ./portfolio_data
COPY ./datasets/tbill_data.csv ./tbill_data.csv

ENV ENV "DOCKER"
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

CMD ["python", "cli.py", \
     "--market-data=portfolio_data", \
     "--risk-free-data=tbill_data.csv", \
     "--n-episodes=100", \
     "--interval=15Min", \
     "--fee=0." \
  ]

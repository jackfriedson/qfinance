FROM python:3.6
WORKDIR /usr/src/qfinance

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./qfinance .
COPY ./datasets/IBM_unadjusted.csv ./stock_data.csv

CMD ["python", "cli.py", "--data-file", "stock_data.csv"]

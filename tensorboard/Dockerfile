FROM python:3.6
WORKDIR /usr/src/tensorboard

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["tensorboard", "--logdir=/usr/share/tensorboard"]

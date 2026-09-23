FROM python:3.8-slim-bullseye

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir awscli

CMD ["python3", "app.py"]
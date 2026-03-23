FROM python:3.12-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

# Running using python ensures it runs exactly as it does on your local machine
CMD ["python", "app.py"]

# docker build -t yt-summarizer .
# docker run -p 5001:5001 --env-file .env yt-summarizer

FROM python:3.12-slim

WORKDIR /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade yt-dlp

COPY . .

CMD ["flask", "run", "--host=0.0.0.0"]


# docker build -t yt-summarizer .
# docker run -p 5000:5000 yt-summarizer

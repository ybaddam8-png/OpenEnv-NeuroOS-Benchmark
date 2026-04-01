FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install openai gymnasium fastapi uvicorn

COPY . /app

EXPOSE 7860
CMD ["python", "app.py"]

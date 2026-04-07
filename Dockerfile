FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies from your list
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project (including the server folder)
COPY . /app

EXPOSE 7860

# Correctly points to app.py inside the server folder!
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
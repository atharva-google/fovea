FROM python:3.11-slim

WORKDIR /app
COPY frontend/requirements.txt /app/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

COPY frontend /app/frontend

ENV PORT=3000
EXPOSE 3000

CMD ["python", "-m", "frontend"]

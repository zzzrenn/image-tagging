FROM python:3.10.11-slim
LABEL description="Image analyzer backend"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y git && pip install -r requirements.txt
EXPOSE 6000
CMD ["python",  "src/app.py"]
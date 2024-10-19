FROM python:3.10.11-slim
LABEL description="Image analyzer frontend"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app/
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
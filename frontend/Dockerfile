FROM python:3.10.11-slim
LABEL description="Image analyzer frontend"
WORKDIR /app
COPY requirements.txt ./
COPY .streamlit ./.streamlit
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["streamlit", "run", "src/app.py", "--server.port", "8080"]
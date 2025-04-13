FROM python:latest
WORKDIR /app_api
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GOOGLE_APPLICATION_CREDENTIALS="/certain-catcher-430110-v2-7beec7335614.json"
EXPOSE 5000
CMD ["python", "model_api.py"]


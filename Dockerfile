FROM python:latest
RUN apt-get update && apt-get install -y \
    curl vim\
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk
    
WORKDIR /app_api
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV GOOGLE_APPLICATION_CREDENTIALS="certain-catcher-430110-v2-7beec7335614.json"
EXPOSE 5000
CMD ["python", "model_api.py"]


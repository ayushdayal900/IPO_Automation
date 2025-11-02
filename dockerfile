FROM python:3.12

WORKDIR /app2

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "your_app.py"]
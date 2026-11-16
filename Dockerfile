FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY employee_attrition_model_from_script.pkl .

EXPOSE 9696

CMD ["python", "predict.py"]

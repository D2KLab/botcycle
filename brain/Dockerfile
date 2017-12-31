FROM python:3
WORKDIR /app
ADD requirements.txt .
ADD nlu_requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -r nlu_requirements.txt
CMD ["python", "main.py"]
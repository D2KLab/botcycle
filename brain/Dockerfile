FROM python:3
WORKDIR /brain
ADD requirements.txt .
ADD nlu_requirements.txt .
RUN apt-get update && apt-get -y install mongodb-clients
RUN pip install -r requirements.txt
RUN pip install -r nlu_requirements.txt
CMD ["python", "main.py"]
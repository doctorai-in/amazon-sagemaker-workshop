FROM python:3.7-buster

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN apt-get update -y && apt-get -y install --no-install-recommends default-jdk
RUN rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install multi-model-server sagemaker-inference sagemaker-training
RUN pip --no-cache-dir install pandas numpy scipy scikit-learn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH="/opt/ml/code:${PATH}"

COPY main.py /opt/ml/code/main.py
COPY train.py /opt/ml/code/train.py
COPY handler.py /opt/ml/code/serving/handler.py

ENTRYPOINT ["python", "/opt/ml/code/main.py"]

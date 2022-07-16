FROM nvidia/cuda:11.4.0-base-ubuntu20.04
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl tzdata software-properties-common
RUN apt-get install unzip
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.10 python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /program

COPY requirements.txt .

RUN python3.10 -m pip install -r requirements.txt

COPY / .

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app" ]

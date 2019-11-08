FROM python:3.7

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git

WORKDIR /face-filters

COPY . /face-filters

RUN pip install opencv-python \
	&& pip3 install numpy \
	&& pip3 install Flask \
	&& pip3 install dlib
	
EXPOSE 8080

ENTRYPOINT  ["python3"]

CMD ["web.py"]

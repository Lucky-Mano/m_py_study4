FROM python:3.7-slim

LABEL maintainer "Lucky <phatbowie@gmail.com>"

ENV DIRNAME=/app

WORKDIR ${DIRNAME}

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install build-essential wget git make curl xz-utils file swig mecab libmecab-dev mecab-ipadic-utf8 -y \
  && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
  && cd mecab-ipadic-neologd \
  && ./bin/install-mecab-ipadic-neologd -n -u -y \
  && pip install -U --no-cache-dir pip setuptools responder mecab2pandas scikit-learn==0.21.3 \
  && apt-get purge build-essential -y

CMD ["python", "app.py"]

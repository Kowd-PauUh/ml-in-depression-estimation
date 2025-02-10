FROM python:3.10-slim-buster

ARG PROJECT_DIR

COPY docker/pythonenv/entrypoint.sh /
COPY docker/pythonenv/requirements.txt /
COPY src/ ${PROJECT_DIR}/src/
COPY data/ ${PROJECT_DIR}/data/

ENV PYTHONPATH=${PROJECT_DIR}

RUN printf '#!/bin/sh\nexit 0\n' > /usr/sbin/policy-rc.d \
 && apt-get update -yqq \
 && apt-get install -y --no-install-recommends curl tar build-essential git-all \
 && apt-get autoremove -yqq --purge \
 && apt-get clean \
 && chmod +x /entrypoint.sh \
 && pip install --no-cache-dir -U -r /requirements.txt

WORKDIR ${PROJECT_DIR}

ENTRYPOINT ["/entrypoint.sh"]

FROM python:3.10-slim-bookworm

ARG PROJECT_DIR
ARG REQUIREMENTS=stable
# REQUIREMENTS can be "soft" or "stable"

ENV PYTHONPATH=${PROJECT_DIR}

# install linux packages
RUN printf '#!/bin/sh\nexit 0\n' > /usr/sbin/policy-rc.d \
  && apt-get update -yqq \
  && apt-get install -y --no-install-recommends \
  curl tar build-essential git-all \
  texlive-full fonts-liberation \
  && apt-get autoremove -yqq --purge \
  && apt-get clean

# add entrypoint
COPY docker/pythonenv/entrypoint.sh /
RUN chmod +x /entrypoint.sh

# install python packages
COPY docker/pythonenv/requirements-${REQUIREMENTS}.txt /requirements.txt
RUN pip install --no-cache-dir -U -r /requirements.txt

COPY src/ ${PROJECT_DIR}/src/
COPY data/ ${PROJECT_DIR}/data/

WORKDIR ${PROJECT_DIR}
ENTRYPOINT ["/entrypoint.sh"]

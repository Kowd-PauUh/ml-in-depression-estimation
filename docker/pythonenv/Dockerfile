FROM python:3.10-slim-bookworm

# install linux packages
# Note: you may remove any package you don't need
#       (e.g. texlive-full for LaTeX support) and add any other
RUN printf '#!/bin/sh\nexit 0\n' > /usr/sbin/policy-rc.d \
  && apt-get update -yqq \
  && apt-get install -y --no-install-recommends \
  curl tar build-essential git-all \
  texlive-full fonts-liberation \
  && apt-get autoremove -yqq --purge \
  && apt-get clean

# install python packages
# Note: REQUIREMENTS arg. can be set to either "soft" or "stable"
ARG REQUIREMENTS=stable
COPY docker/pythonenv/requirements-${REQUIREMENTS}.txt /requirements.txt
RUN pip install --no-cache-dir -U -r /requirements.txt

# add entrypoint
COPY docker/pythonenv/entrypoint.sh /
RUN chmod +x /entrypoint.sh

# set pythonpath
ARG PROJECT_DIR
ENV PYTHONPATH=${PROJECT_DIR}

# copy files into container
COPY src/ ${PROJECT_DIR}/src/
COPY data/ ${PROJECT_DIR}/data/

WORKDIR ${PROJECT_DIR}
ENTRYPOINT ["/entrypoint.sh"]

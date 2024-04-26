#!/bin/bash

if [ -e "/init/init.sh" ]; then
  /init/init.sh
fi

if [ -z "$1" ]; then
  sleep infinity
elif [ "$1" == "app" ]; then
  streamlit run src/application/main.py --server.address=0.0.0.0 --server.enableXsrfProtection=false
elif [ "$1" == "notebook" ]; then
  jupyter notebook --port=8000 --ip=0.0.0.0 --NotebookApp.token='' --allow-root
else
  exec $@
fi

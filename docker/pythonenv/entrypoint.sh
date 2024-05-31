#!/bin/bash

if [ -z "$1" ]; then
  sleep infinity
elif [ "$1" == "notebook" ]; then
  jupyter notebook --port=8000 --ip=0.0.0.0 --NotebookApp.token='' --allow-root
else
  exec $@
fi

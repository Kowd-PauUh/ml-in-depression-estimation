#!/bin/bash

if [ -z "$1" ]; then
  sleep infinity
else
  exec $@
fi

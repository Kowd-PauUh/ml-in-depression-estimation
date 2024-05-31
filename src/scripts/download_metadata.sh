#!/bin/bash

base_url="https://dcapswoz.ict.usc.edu/wwwedaic/labels/"

mkdir -p data/raw_data
cd data/raw_data

file_list=$(curl -s $base_url | grep -oP '(?<=href=")[^"]*\.csv')

for file in $file_list; do
    file_url="${base_url}${file}"
    echo "Downloading $file_url"
    curl -O $file_url
done

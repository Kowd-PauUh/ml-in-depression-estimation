#!/bin/bash

base_url="https://dcapswoz.ict.usc.edu/wwwedaic/data/"

mkdir -p data/raw_data
cd data/raw_data

file_list=$(curl -s $base_url | grep -oP '(?<=href=")[^"]*\.tar')

for file in $file_list; do
    file_url="${base_url}${file}"
    echo "Downloading $file_url"
    curl -O $file_url
    tar -xf $file
    rm $file
    cd $(echo $file | sed 's/\.tar$//')
    rm -r features/
    cd ..
done

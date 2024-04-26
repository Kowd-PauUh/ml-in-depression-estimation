#!/bin/bash

# Mounting DB in postgres container
shopt -s extglob

postgres_url=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT
database=$POSTGRES_DB_NAME

echo -e $postgres_url
echo -e $database
cat /postgres/$database.sql

# Checking if database is up
until psql $postgres_url -c "\conninfo" > /dev/null 2>&1; do
  echo "Postgres metadata db is unreachable - sleeping..."
  sleep 10
done

echo -e "\nPostgres metadata is up"
echo -e "\nCreating databases and schemas"

# Creating databases and schemas
if [ -z $(psql $postgres_url -Atc "\list "$database";") ]; then
  psql $postgres_url -a -c "create database "$database";"
  psql $postgres_url/$database -a -c "CREATE EXTENSION IF NOT EXISTS periods version '1.1' cascade;"
  psql $postgres_url/$database -a -f /postgres/$database.sql
  echo -e "Created database "$database""
fi
echo "Database "$database" already exists"

if [ -z "$1" ]; then
  sleep infinity
elif [ "$1" == "notebook" ]; then
  jupyter notebook --port=8000 --ip=0.0.0.0 --NotebookApp.token='' --allow-root
else
  exec $@
fi

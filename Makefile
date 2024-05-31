#!make

export 

PROJECT_NAME = $(notdir $(PWD))
PROJECT_DIR = $(PWD)

env:
	echo "PROJECT_DIR=$(PROJECT_DIR)" >> docker/.env
	echo "PROJECT_NAME=$(PROJECT_NAME)" >> docker/.env

clean:
	docker-compose -f docker/docker-compose.yml -p $(PROJECT_NAME) down --remove-orphans

build:
	make env && docker-compose -f docker/docker-compose.yml -p $(PROJECT_NAME) build

start:
	docker-compose -f docker/docker-compose.yml -p $(PROJECT_NAME) up -d

restart:
	make stop && make build && make start

stop:
	docker-compose -f docker/docker-compose.yml -p $(PROJECT_NAME) stop

shell: 
	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c "/entrypoint.sh bash"

# app:
# 	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c "/entrypoint.sh app"

notebook:
	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c "/entrypoint.sh notebook"

config: FORCE
	make env && docker-compose -f docker/docker-compose.yml config

FORCE: ;
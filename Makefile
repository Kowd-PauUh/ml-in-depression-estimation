# "soft" or "stable"
REQUIREMENTS = stable

PROJECT_DIR = $(PWD)
PROJECT_NAME = $(shell echo $(notdir $(PWD)) | tr A-Z a-z)

# prepare docker/.env file
env:
	echo "PROJECT_DIR=$(PROJECT_DIR)" > docker/.env
	echo "PROJECT_NAME=$(PROJECT_NAME)" >> docker/.env
	echo "REQUIREMENTS=$(REQUIREMENTS)" >> docker/.env

# (i) stop container, (ii) remove container, networks, and any orphaned containers
clean:
	docker compose -f docker/docker-compose.yml -p $(PROJECT_NAME) down --remove-orphans

# build image
build:
	make env && docker compose -f docker/docker-compose.yml -p $(PROJECT_NAME) build

# start container with allocated GPU
start:
	docker compose -f docker/docker-compose.yml -p $(PROJECT_NAME) up -d

# start container without GPU
start-cpu:
	docker compose -f docker/docker-compose.cpu.yml -p $(PROJECT_NAME) up -d

# rebuild image and recreate container
restart:
	make stop && make build && make start

# stop container
stop:
	docker compose -f docker/docker-compose.yml -p $(PROJECT_NAME) stop

# enter container shell
shell: 
	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c "/entrypoint.sh bash"

# start mlflow server at http://0.0.0.0:1234/ inside container
# Note: mlflow server by default is mapped to the port 8502
# 		on your machine, as specified in docker/docker-compose.yml
mlflow:
	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c \
	"/entrypoint.sh mlflow server --host=0.0.0.0 --port=1234 \
	--backend-store-uri $(PROJECT_DIR)/data/mlruns \
	--artifacts-destination $(PROJECT_DIR)/data/mlartifacts"

# start jupyterlab server at http://0.0.0.0:8000/ inside container
# Note: jupyterlab server by default is mapped to the port 8501
# 		on your machine, as specified in docker/docker-compose.yml
notebook:
	docker exec -it ${PROJECT_NAME}-pythonenv /bin/sh -c \
	"/entrypoint.sh jupyter lab --port=8000 --ip=0.0.0.0 --NotebookApp.token='' --allow-root"

# print docker compose configuration to console
config: FORCE
	make env && docker compose -f docker/docker-compose.yml config

FORCE: ;

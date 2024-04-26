## Setup
- ensure you have defined `POSTGRES_HOST` variable in your environment. It can be your localhost. 
- run `make build && make start` in terminal
- run `make shell` to enter the container shell
- inside the shell run `python3 src/scripts/load_postgres.py`
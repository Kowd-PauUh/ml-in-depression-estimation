## Setup
- Ensure you have defined `POSTGRES_HOST` variable in your environment. It can be your localhost. 
- Run `make build && make start` in terminal
- Run `make shell` to enter the container shell
- Inside the shell run `python3 src/scripts/load_postgres.py` to load data to the postgres
- Exit the shell using `exit` and then run `make notebook` in the terminal to open jupyterlab
- Navigate to the `notebooks/marketing-campaign-modelling.ipynb` in the file browser, open it and run through the cells
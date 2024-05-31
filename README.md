## Getting started
- Run `make build && make start` in terminal
- To enter the container shell run `make shell` in the terminal
- To open jupyterlab run `make notebook` in the terminal
- Explore other available `make` commands (targets) in `Makefile`

## Project tree overview
The project is organized as following
```
project/                                              
├── data/                    # data
│
├── docker/                  # services
│   ├── service_x/
│   │   ├── Dockerfile
│   │   └── ...
│   └── docker-compose.yml
│
├── src/                     # code modules
│   ├── module_x/
│   │   ├── ...
│   │   └── __init__.py
│   └── __init__.py
|
├── .dockerignore
├── .gitignore
├── Makefile                 # tasks automation
└── README.md
```

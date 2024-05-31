## Getting started
- Run `make build && make start` in terminal
- To enter the container shell run `make shell` in the terminal
- To open jupyterlab run `make notebook` in the terminal
- Explore other available `make` commands (targets) in `Makefile`

## Data download
Data used in this repository is a part of the Extended DAIC-WOZ Database [1, 2] and can only be posessed after completing End-User License Agreement. For detailed information email `daicwoz@ict.usc.edu`. After you've obtained an access to the database you can run `src/scripts/download_data.sh` and `src/scripts/download_metadata.sh` scripts from terminal.

[1] Gratch, Jonathan, et al. "The distress analysis interview corpus of human and computer interviews." LREC. 2014.

[2] Ringeval, Fabien, et al. "AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition." Proceedings of the 9th International on Audio/visual Emotion Challenge and Workshop. 2019.

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

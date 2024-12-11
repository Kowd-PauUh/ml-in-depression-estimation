## Getting started
- Run `make build && make start` in terminal
- To enter the container shell run `make shell` in the terminal
- To open jupyterlab run `make notebook` in the terminal
- Explore other available `make` commands (targets) in `Makefile`

## Data download
Data used in this repository is a part of the Extended DAIC-WOZ Database [1, 2] and can only be posessed after completing End-User License Agreement. For detailed information email `daicwoz@ict.usc.edu`. After you've obtained an access to the database you can run `src/scripts/data/download_data.sh` and `src/scripts/data/download_metadata.sh` scripts from terminal.

Voice recording example `data/OSR_us_000_0030_8k.wav` used in notebooks is taken from Open Speech Repository.

[1] Gratch, Jonathan, et al. "The distress analysis interview corpus of human and computer interviews." LREC. 2014.

[2] Ringeval, Fabien, et al. "AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition." Proceedings of the 9th International on Audio/visual Emotion Challenge and Workshop. 2019.

## Features extraction

For the baseline purposes we implement waveform features extraction functions which can be found in consecutive scripts in `src/waveform/features_extraction` directory and be applied to the waveform tensor directly. We also use features implemented in [DisVoice](https://gitlab.com/Kowd-PauUh/disvoice)<sup>1</sup> package, which are extracted via `src/scripts/extract_disvoice_features.py` script execution, for example: 
```
python3 src/scripts/extract_disvoice_features.py --features_extractor_cls_name=Articulation
```
Allowed arguments are: `features_extractor_cls_name`, `dataset_path`, `split_df_path`, `features_save_dir`, `match_split_df_on`, `grouping_column_name`, `split_column_name`, `splits`, `downsample_to`.
To extract these features from code, follow notebooks in the [DisVoice](https://gitlab.com/Kowd-PauUh/disvoice) repository.

For the CNN training we compute MEL-spectrograms. Spectrograms extraction is documented in the `src/waveworm/transformation/mel_spectrogram.py` script.

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

---
<sup>1</sup> Important note: This is a revised DisVoice version with fixed bugs and resolved dependencies conflicts and it is not supported by the author of the [original](https://github.com/jcvasquezc/DisVoice/tree/master/disvoice) package.

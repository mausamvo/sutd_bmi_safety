# new_ml_code

Per-subject sEMG gesture classification pipeline: baseline vs attack recordings.

- `src/config.py`: all settings. Set `SUBJECT` to choose which subject to run.
- `src/data.py`: file-level train / validation / test split (never by window); attack files are held out.
- `src/dataset.py`, `src/features.py`: resampling to a common 120 Hz, windowing, CWT scalograms and hand-crafted features.
- `src/model.py`, `src/train.py`: EMGNet reconstruction (Chen et al., Sensors 2020) and the EMGNetFusion network.
- `src/classical.py`: RandomForest / ExtraTrees search with file-grouped cross-validation.
- `src/run_pipeline.py`: runs everything and writes results to `outputs/<SUBJECT>/`.
- `analysis/rate_experiment.py`: compares sample rates on the same recordings.

## Running

```bash
pip install -r requirements.txt
python3 -m src.run_pipeline
```

The data is not in this repository. `DATA_ROOT` in `src/config.py` must point to a folder laid out as
`<subject>/<session>/<epoch>_<session>_<gesture>_set<n>_adc.csv`.

`SEMG_TRAIN_ON=matched python3 -m src.run_pipeline` trains only on the baseline sessions recorded on the same setup as the attacks.

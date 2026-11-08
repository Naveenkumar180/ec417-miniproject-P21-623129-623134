# Data

- Dataset and version: ECG Heartbeat Categorization Dataset (MIT-BIH arrhythmia beats, segmented), CSV copy `mitbih_train.csv` / `mitbih_test.csv`. **Fill in the exact Kaggle slug/version you used** (script default: `shayanfazeli/heartbeat`).
- Official source URL: Kaggle dataset page (fill in); underlying data: MIT-BIH Arrhythmia Database on PhysioNet. Preprocessing paper: Kachuee, Fazeli, Sarrafzadeh, "ECG Heartbeat Classification: A Deep Transferable Representation", IEEE ICHI 2018, doi:10.1109/ICHI.2018.00092.
- Access date: **TODO (YYYY-MM-DD)**
- License / terms: **TODO – copy from the Kaggle dataset card and from PhysioNet (Open Data Commons Attribution)**. Not redistributed in this repo.
- Required citation: Kachuee et al. 2018 (above); Moody & Mark, "The impact of the MIT-BIH Arrhythmia Database", IEEE EMB Mag. 2001; Goldberger et al., PhysioNet, Circulation 2000.
- Files used: `mitbih_train.csv` (87,554 beats), `mitbih_test.csv` (21,892 beats); 109,446 in total. SHA-256 sums are written to `data/raw/SHA256SUMS.txt` by the download script — paste them here.
- Download/preparation command: `python scripts/get_data.py`
- Preprocessing and filtering: none beyond what the dataset provides. Each beat is 187 samples at 125 Hz, already amplitude-normalised to [0, 1] and zero-padded by the dataset authors, so no scaler is fitted. No rows are removed.
- Label mapping: 0 = N (normal), 1 = S (supraventricular ectopic), 2 = V (ventricular ectopic), 3 = F (fusion), 4 = Q (unknown/paced).
- Train/validation/test split: test = official `mitbih_test.csv`. Validation = stratified 10 % of `mitbih_train.csv` (`split_seed = 42`), remainder = training. Same split for every model.
- Random seed(s): split seed 42; model seeds 0, 1, 2.
- Known limitations, leakage risks, privacy concerns:
  - **Beat-level split, no patient IDs**: beats from the same patient appear in train, validation and test, so scores overestimate performance on new patients. Do not claim patient-independent generalisation.
  - Exact duplicate beats between train and test are counted in `notebooks/01_eda.ipynb`.
  - Severe class imbalance (class N ≈ 83 %): accuracy is dominated by N, so macro-F1 and per-class recall are the headline metrics.
  - De-identified public research data; educational project, **not a clinical tool**.

To reproduce:
1. `pip install -r requirements.txt`
2. `python scripts/get_data.py`
3. Run the preparation/training command documented in the project README.

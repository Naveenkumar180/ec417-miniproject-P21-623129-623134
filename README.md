# P21 – ECG Heartbeat Classification: LSTM, BiLSTM and CNN-LSTM vs a Random-Forest Baseline

EC-417 Introduction to Machine Intelligence, NIT Andhra Pradesh · Team: roll nos. 623129, 623134

> **Headline result:** _TODO – one sentence, e.g. "CNN-LSTM with class weights reaches test macro-F1 0.xx ± 0.xx vs 0.xx ± 0.xx for the random forest (3 seeds)"._

## 1. Problem
Classify each ECG heartbeat (187 samples at 125 Hz) into one of 5 classes: N (normal), S (supraventricular), V (ventricular), F (fusion), Q (unknown).
Arrhythmia screening cares about the rare classes, but ~83 % of beats are class N, so **accuracy is reported but is not the headline metric**. Headline metric: **macro-F1** (every class counts equally); secondary: per-class recall (a missed arrhythmia is the costly error).
Educational project, **not a clinical tool**.

## 2. Data
Kachuee et al. (2018) segmentation of MIT-BIH, Kaggle CSV copy: 87,554 train + 21,892 test = 109,446 beats. Details, licence, checksums and limitations: [`data/README.md`](data/README.md).
Split: official test file untouched; validation = stratified 10 % of the official training file (seed 42). _TODO: paste class-balance table from `notebooks/01_eda.ipynb`._
**The split is beat-level (no patient IDs), so the same patient can appear in train and test; scores overestimate performance on new patients.**

## 3. Method
| Role | Model |
|---|---|
| Trivial baseline | Predict the majority class |
| Baseline | Random forest (200 trees) on the raw 187 samples |
| Main model | LSTM and BiLSTM (hidden 64) on the beat as a length-187 sequence |
| Improvement | CNN-LSTM (2 conv blocks → LSTM) **with** class-weighted cross-entropy |

The brief's improvement bundles two changes (conv front end, class weights), so we also run two ablations — BiLSTM + class weights, CNN-LSTM without weights — to isolate each. All neural models: Adam (lr 1e-3), batch 256, up to 30 epochs, early stopping on validation macro-F1 (patience 5), gradient clipping 1.0.

## 4. Results (test set, mean ± std over 3 seeds, reported once at the end)
_TODO: paste `results/results_table.md` after the real runs._

Per-class recall: `results/per_class_recall.md`. Figures: `results/figures/` (confusion matrices, per-class recall).

## 5. Error analysis
_TODO: which classes are confused (expect S↔N, F↔V), examples of misclassified beats over the class-average waveform, why._

## 6. Limitations and ethics
- Beat-level split → patient leakage; no claim of new-patient generalisation.
- Class N dominates; rare-class recall has wide uncertainty (class F has few test beats).
- Not validated for clinical use; do not use for diagnosis.

## 7. Reproduce
Hardware used: _TODO (e.g. Colab T4)_ · total runtime: _TODO_
```bash
pip install -r requirements.txt
python scripts/get_data.py                       # needs Kaggle API token
bash scripts/run_all.sh                          # 7 configs x 3 seeds -> results/metrics.csv
python -m src.evaluate                           # table + figures
```
Or open `notebooks/02_experiments.ipynb` in Colab and "Run all".

## 8. AI-use statement
_TODO – fill in honestly._ Starting point: the repository scaffold, training/evaluation code and README skeleton were drafted with Claude (Anthropic). We must be able to explain every line at the viva; list here what we personally re-read, ran, changed and verified.

## 9. References
- Kachuee, Fazeli, Sarrafzadeh. ECG Heartbeat Classification: A Deep Transferable Representation. IEEE ICHI 2018. doi:10.1109/ICHI.2018.00092
- Moody & Mark. The impact of the MIT-BIH Arrhythmia Database. IEEE EMB Magazine 2001.
- Goldberger et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000.

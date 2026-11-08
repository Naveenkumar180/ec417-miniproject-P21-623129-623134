# Report outline (4–6 pages, same order as README, with reasoning)

1. Framing – why macro-F1 + per-class recall, why accuracy misleads (83 % N)
2. Data & split – class table, beat-level split and what it means for claims, duplicate check result
3. Baselines – majority, RF: what they get and why RF on raw samples is strong
4. Main models – LSTM vs BiLSTM: parameters, time, F1; learning curves (val F1 per epoch)
5. Improvement – CNN-LSTM + class weights, plus the two ablations: which ingredient helped?
6. Evaluation – mean ± std over 3 seeds, confusion matrices, per-class recall
7. Error analysis – misclassified beats over class-average waveform (stretch)
8. Limitations / next steps – patient-wise split (needs patient IDs from raw MIT-BIH), oversampling vs weighting
9. AI-use statement

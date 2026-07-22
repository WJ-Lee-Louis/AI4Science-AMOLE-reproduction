# Zero-shot cross-modal retrieval (@4)

Values are five-trial mean accuracy (%) with population standard deviation.

| Method | Description<br>Given Molecule @4 | Description<br>Given Text @4 | Pharmacodynamics<br>Given Molecule @4 | Pharmacodynamics<br>Given Text @4 | ATC<br>Given Molecule @4 | ATC<br>Given Text @4 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (alpha=1.0) | 91.39 +/- 0.65 | 90.52 +/- 0.60 | 85.03 +/- 0.69 | 83.38 +/- 0.67 | 71.67 +/- 0.16 | 68.33 +/- 0.49 |
| curriculum (alpha=1.0) | 92.10 +/- 0.62 | 91.11 +/- 0.70 | 85.05 +/- 0.58 | 83.36 +/- 0.66 | 70.57 +/- 0.70 | 68.67 +/- 0.38 |
| stratified (alpha=1.0) | 91.68 +/- 0.53 | 90.17 +/- 0.63 | 84.92 +/- 0.73 | 82.97 +/- 0.56 | 70.66 +/- 0.49 | 68.14 +/- 0.25 |
| curriculum (alpha=2.0) | 91.44 +/- 0.58 | 90.49 +/- 0.64 | 85.33 +/- 0.37 | 83.54 +/- 0.87 | 72.01 +/- 0.41 | 69.35 +/- 0.36 |

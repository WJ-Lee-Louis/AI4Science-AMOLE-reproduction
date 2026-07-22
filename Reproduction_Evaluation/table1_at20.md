# Zero-shot cross-modal retrieval (@20)

Values are five-trial mean accuracy (%) with population standard deviation.

| Method | Description<br>Given Molecule @20 | Description<br>Given Text @20 | Pharmacodynamics<br>Given Molecule @20 | Pharmacodynamics<br>Given Text @20 | ATC<br>Given Molecule @20 | ATC<br>Given Text @20 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (alpha=1.0) | 79.58 +/- 0.74 | 78.35 +/- 0.64 | 68.44 +/- 0.91 | 65.93 +/- 0.58 | 47.07 +/- 0.18 | 43.38 +/- 0.71 |
| curriculum (alpha=1.0) | 80.07 +/- 0.27 | 78.94 +/- 0.65 | 68.54 +/- 0.92 | 66.69 +/- 0.71 | 45.99 +/- 0.50 | 43.07 +/- 0.29 |
| stratified (alpha=1.0) | 79.15 +/- 0.75 | 77.97 +/- 0.52 | 68.06 +/- 0.75 | 66.11 +/- 0.79 | 46.17 +/- 0.31 | 42.86 +/- 0.46 |
| curriculum (alpha=2.0) | 79.38 +/- 0.56 | 78.11 +/- 0.45 | 68.48 +/- 1.07 | 65.87 +/- 0.80 | 47.88 +/- 0.31 | 44.41 +/- 0.75 |

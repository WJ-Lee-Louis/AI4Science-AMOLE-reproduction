# Zero-shot cross-modal retrieval (@10)

Values are five-trial mean accuracy (%) with population standard deviation.

| Method | Description<br>Given Molecule @10 | Description<br>Given Text @10 | Pharmacodynamics<br>Given Molecule @10 | Pharmacodynamics<br>Given Text @10 | ATC<br>Given Molecule @10 | ATC<br>Given Text @10 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (alpha=1.0) | 84.99 +/- 0.52 | 83.64 +/- 0.64 | 75.64 +/- 0.48 | 73.31 +/- 0.47 | 56.52 +/- 0.38 | 52.70 +/- 0.46 |
| baseline (alpha=2.0) | 84.73 +/- 0.75 | 83.28 +/- 0.39 | 75.76 +/- 0.21 | 73.43 +/- 0.48 | 56.27 +/- 0.38 | 51.52 +/- 0.46 |
| curriculum (alpha=1.0) | 85.63 +/- 0.39 | 84.23 +/- 0.67 | 75.84 +/- 0.81 | 73.59 +/- 0.76 | 55.40 +/- 0.36 | 52.80 +/- 0.27 |
| curriculum (alpha=2.0) | 84.78 +/- 0.81 | 83.41 +/- 0.37 | 76.50 +/- 0.40 | 73.69 +/- 0.78 | 57.27 +/- 0.63 | 54.15 +/- 0.50 |
| stratified (alpha=1.0) | 84.70 +/- 0.63 | 82.84 +/- 0.79 | 76.16 +/- 0.72 | 73.13 +/- 0.68 | 55.62 +/- 0.20 | 52.70 +/- 0.24 |
| stratified (alpha=2.0) | 84.45 +/- 0.54 | 82.79 +/- 0.60 | 75.88 +/- 0.47 | 72.48 +/- 0.51 | 55.90 +/- 0.52 | 52.16 +/- 0.58 |

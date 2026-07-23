# Hyperparameter specifications for AMOLE reproduction pretraining

| Hyperparameter | Value |
| --- | --- |
| Training epochs | 20 |
| Learning rate for text encoder f_{text} | 1 \times 10^{-5} |
| Learning rate for molecule encoder f_{mol} | 1 \times 10^{-5} |
| Temperature for pseudo-label \tau_1 | 0.1 |
| Temperature for model prediction \tau_2 | 0.1 |
| Maximum number of similar molecules k | 50 |
| Replacement probability p | 0.5 |
| Weight of expertise reconstruction loss \alpha | {1.0, 2.0} |
| Global batch size | 30 (10 per GPU × 3 GPUs) |
| Maximum text sequence length | 512 tokens |

The table reports the common pretraining settings and the evaluated ER-loss weights.

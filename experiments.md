A few baseline params we will use for now:
```
DATA_SHAPES = [
    (3, 32, 32),
    (8, 8, 8),
    (8, 8, 8),
    (8, 8, 8),
    (8, 8, 8),
    128,
    100,
]
BATCH_SIZE = 64
LR = 0.0001
DEGREE = 2
EPOCHS = 25

```
### PARAM_COUNT = 13k
1. 3.5%. Linear interpolation, FuzzyNAND aggregation.
2. 4.0%. Fourier interpolation, FuzzyNAND aggregation.
### PARAM_COUNT = 38k, DEGREE = 4
3. 13.67%. Linear interpolation, linear combination aggregation.
4. 10.51%. Fourier interpolation, linear combination aggregation. Much less monotonic improvement, trained 10x slower in wall-clock time too. Will be abandoning Fourier interp going forward.
### PARAM_COUNT = 74k, DEGREE = 8 (DOUBLED DEGREE)
5. 18.36%. Linear, linear. 25 epochs.
### PARAM_COUNT = 73k, DEGREE = 4 (DOUBLED MID-LAYER COUNT)
6. 14.08%. 25 epochs.
— **Increased batch size (256) and training time (up to 100 epochs)**
7. 22.94%, DEGREE=8, 74k params.
8. 19.375, DEGREE=4, 73k params. (double mid-layer count)

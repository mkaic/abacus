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

```
### PARAM_COUNT = 13k
1. 3.5%. Linear interpolation, FuzzyNAND aggregation.
2. 4.0%. Fourier interpolation, FuzzyNAND aggregation.
### PARAM_COUNT = 13k
3. X.X%. Linear interpolation, linear combination aggregation.
4. X.X%. Fourier interpolation, linear combination aggregation.

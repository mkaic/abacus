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
8. 19.37%, DEGREE=4, 73k params. (double mid-layer count)
9. 19.65%, DEGREE=4, 84k params. (mid-block lookbehind 1 -> 2)
10. 18.18%, Degree=4, 84k params. (mid-block lookbehind 2 -> 4)
11. 18.05%, Degree=4, 84k params. (mid-block lookbehind 4 -> 8)
12. 16.82%. 71k params. same as (7) but without ReLU or biases.
13. 17.64%. 74k params. same as (7) but with only biases, no ReLU.
14. 22.53%. 71k params. same as (7) but with only ReLU, no biases.
15. 22.91%. 71k params. same as (14) but with LeakyReLU(0.1).
16. 22.56%. 71k params. same as (14) but with GELU.

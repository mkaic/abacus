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
17. 15.63%. 66k params. degree 4, with params *roughly* doubled by widening the 4 mid-block layers to (10,10,10)
18. 22.67%. 67k params. degree 16, with params made equal by shrinking the 4 mid-block layers to (6, 6, 6)
19. 12.83%. 80k params. degree 10. 1x (5,5,5,5) layer. 10 x (5,5,5) layers.
20. CUDA LAUNCH BLOCKING ERROR. 90k params. same as (19) but with lookbehind 5.
21. 23.61%. 93k params. degree 8. 1x (6,6,6,6), 6x (6,6,6), lookbehind 6.
22. 20.70%. 34k params. degree 6. 6x (6,6,6), lookbehind 1.
23. 17.76%. 38k params. same as (22) but with lookbehind 3. Lookbehind truly doesn't seem to help at all.
24. 9.47%. 34k params. same as (22) but with LR=1e-4 instead of 1e-3. Low LR is unnecessary.
25. xx.xx%. 34k params. same as (22) but with LR=1e-2 instead of 1e-3.
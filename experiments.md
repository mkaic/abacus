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
25. 16.96%. 34k params. same as (22) but with LR=1e-2 instead of 1e-3. High LR is harmful
26. 19.93%. 34k params. same as (22) but with batch size 512 instead of 256. Bigger batch size did nothing.

New config time:
```
INPUT_SHAPES=[(3, 32, 32)], MID_BLOCK_SHAPES=[(4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4)], OUTPUT_SHAPES=[(100,)], DEGREE=4, BATCH_SIZE=256, LR=0.001, INTERPOLATOR=<class 'abacus.src.interpolators.LinearInterpolator'>, AGGREGATOR=<class 'abacus.src.aggregators.LinearCombination'>, LOOKBEHIND=1, EPOCHS=100, DATA_DEPENDENT=True
Initialized SparseAbacusModel with 69,776 total trainable parameters.
```
27. 13.58%. 70k params. Trying out full data-dependence on a narrow model (since DD balloons param counts.)

```
INPUT_SHAPES=[(3, 32, 32)], MID_BLOCK_SHAPES=[(6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6), (6, 6, 6)], OUTPUT_SHAPES=[(100,)], DEGREE=6, BATCH_SIZE=256, LR=0.001, INTERPOLATOR=<class 'abacus.src.interpolators.LinearInterpolator'>, AGGREGATOR=<class 'abacus.src.aggregators.LinearCombination'>, LOOKBEHIND=1, EPOCHS=100, DATA_DEPENDENT=[False, False, False, False, False, False, True]
Initialized SparseAbacusModel with 74,904 total trainable parameters.
```
28. 22.01% with Data Dependency on only the final layer.
29. 14.62%. 49k params. Adding a few 100-neuron layers to the end of the model and giving only them data-dependency.
30. BAD. same as (29) but with torch.clamp instead of torch.sigmoid acting to limit the outputs of sample points predictors.
31. 15.18%. 34k params. Let's go the opposite direction. Instead of clamping all sample points, let's sigmoid all sample points all the time and get rid of the clamp_params calls. Returning to the config of (22) for this. This also means sample points should be initialized in mean-zero way.
32. 21.37%. 270k params. none of my ideas have worked lmao. time to make number bigger and see what happens. 8x (8,8,8) layers with degree 16. update: it did not work.

new config:
```
INPUT_SHAPES=[(3, 32, 32)], MID_BLOCK_SHAPES=[(8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8), (8, 8)], OUTPUT_SHAPES=[(100,)], DEGREE=8, BATCH_SIZE=256, LR=0.001, INTERPOLATOR=<class 'abacus.src.interpolators.FourierInterpolator'>, AGGREGATOR=<class 'abacus.src.aggregators.LinearCombination'>, LOOKBEHIND=1, EPOCHS=100, DATA_DEPENDENT=[False, False, False, False, False, False, False, False, False]
Initialized SparseAbacusModel with 15,200 total trainable parameters.
```
33. 14.28%. 15k params. trying out a tiny fourier-interp based model again bc I'm hitting a wall with Linear Interp.
34. 15.20%. 15k params. same as (33) but with Linear Interp as a fair side-by-side comparison. urgh. giving up on Fourier again now.
35. 16.16%. 15k params. same as (33) but I'm logging the weights every epoch now. Also re-enabled sample points clipping, which I had forgotten was disabled at one point.
36. 16.74%. 9k params. same as (33) but only 4 (8,8) mid-block layers this time. a truly miniscule model. good god it outperformed the one twice its size, what the hell is wrong with my architecture's grad flow lmao??
37. 9.58%. 3k params. Let's take it to the logical extreme. No mid-block layers at all. Literally just an output layer directly sampling and aggregating from the input image. WHY DOES IT GET TO 10% ACC AAAAAA.
38. 14.87% 13k params. okay what happens if i up the degree from 8 to 32 and still have zero mid blocks?
39. 17.09%. 51k params. let's up the degree to 128 now. i hate that this is working this well.
40. 17.05%. 39k params. degree back to 32. added a 256 midblock. did nothing.
41. 20.32%. 34k params. Realized that the real bottleneck with fourier interp is actually the *first layer* bc it's 32x32x3 sinusoids that have to be evaluated and that's non-negligible. Smaller midblock fourier layers are much cheaper and so for this run I am using LinearInterp for the first layer and FourierInterp for all subsequent layers. Same structure and param count as (22).
42. 21.37%. 34k params. Same as (22) but with LeakyReLU negative slope of 0.5.
43. 2.xx%. 34k params. Same as (42) but with ±1 neuron-variance gaussian noise added to sample points.
44. 20.74%. 34k params. Same as (42) but with the noise has sigma 0.1 instead of 1.
45. 17.xx%. 34k params. Same as (42) with noise sigma 0.2 instead of 1.
46. 2.xx%. 206k params. (16x16) x 16, degree 16, noise sigma 0.2, batch_size 512.
47. like 20%. 34k params. 10 x (10,10), degree 10. residual connections in all midblocks.
48. MEH%. 34k params. same as (47) but with noise sigma 0.2 added back.
49. 24.79%. 81k params. (5,5,5) x 10, degree 15. residuals!
50. 25.24%. 82k params. same as (49) but with biases added back in to the aggregator layers.
51. 24.20%. 84k params. (4,4,4) x 12, degree 24.
52. 25.06%. 68k params. (6,6,6) x 6, degree 12
53. 24.06%. 77k params. same as (52) but with lookbehind=6. Lookbehind is doomed it seems.
54. 24.90%. 87k params. (4,4,4,4) x 4, degree 16
55. 19.07%. 65k params. (5,5,5,5) x 5, degree 4, lookbehind=1
56. 17.60%. 34k params. Same as (55) but with degree 2.
57. 17.xx#. stopped early. 181k params. (7,7,7,7) x 7, degree 2.
58. 17-ish%. stopped early as asymptote was obvious. 25k params. (3**5) x 8, degree 2.
59. 17-ish%. stopped early. 23k params. (4**4) x 8. degree 2. LinearCombo agg.
60. 21.70%. 23k params. same as (59) but trying out FuzzyNAND as an activation function in LinearCombo. the dream for fuzzy logic degree-2 sparse networks has not died. yet. HELL YEAH IT OUTPERFORMED LeakyReLU
61. 9.xx%. 34k params. same as (60) but with degree-3 because I'm curious how well that will play with the FuzzyNAND activation. got stuck.
62. 9.xx%. 44k params. what about degree=4? is it an even-odd thing maybe? result: no, degree=2 is just optimal for FuzzyNAND.
63. 9.xx%, 46k params. let's go deeper. 8 layers -> 16 layers. otherwise same as (60). result: got stuck.
64. 10.xx%, 113k params. same as (60) but with midblock layer size (6**4) now.
65. 9.xx%, 50k params. (4**5) x 4.
NOTE: accidentally turned on input clamping prior to FuzzyNAND and forgot, that was disastrous! it absolutely cripples gradient flow I think.
66. 12.xx%, 17k params. Same as (59) but without weights and biases, only trainable params are sample points and activation is FuzzyNAND *WITHOUT* input clamping. result: stagnated at 12%. maybe adding leakyrelu after fuzzynand could help?
67. NIXED, 25k params. Same as (66) but with degree 3 just to sanity check and make sure it wasn't the clamping that borked (61). result: lags behind degree 2 like before.
68. 21.xx%, 46k params. a clone of (63) to see if *that* was borked by clamping. result: IT WAS. deep networks can actually train, yay!
69 (nice). xx.xx%, 68k params. (32,32) x 16, no weights, no biases.
70. 18.xx%, 60k params. (32,32) x 8, WITH weights and biases.
71. 21.xx%, 25k params. (4**4) x 8, weighted and biased, with an additional parameter per neuron to allow for weighting between FuzzyNAND and FuzzyNOR. No clear benefit.
72. NIXED, TOO SLOW, 85k params. (3**6) x 8, weighted/biased FuzzyNAND. Degree 2.
73. 24.26%, 113k params. (10**3) x 4, (8**3) x 28.
74. 13.64%, 2.5k params. (2**5) x 2. Binary tree subcase of linear interpolation is more compute and memory-performant. Also hella parameter-efficient and way simpler to implement! Does require padding the input to
75. 23.24%, 42k params. (2**8) x 8. 
76. NIXED%, 106k params. (2**11), (2**10), (2**9), (2**8), (2**7) x 3. bottleneck arch. worrried bc it can't have skip-cons in the early layers due to shape diffs. TOO SLOW.
77. 17.xx%, 42k params. same as (75) but with LinearCombination instead of LinearFuzzyNAND.
78. 20.xx%, 48k params. same as (77) but with layer size (2**5) instead of (2**8), and degree 16 instead of 2. 1/8 the layer size, 8x the degree.
79. xx.xx%, 38k params. (2**8) x 16, degree 2. deep network.

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).

# Experiments
* Image inductive bias seems not to do much tbh.
* Baseline acc,  
```
1. data_shapes = [(3, 32, 32), (128, 128), (1024,), (512,), (100,)] # PARAMS: 103k, ACC: ~5%
2. data_shapes = [(3, 32, 32), (128, 128), (64,64), (32,32), (16,16), (100,)] # PARAMS: 120k, ACC: ~7%
3. SAME data_shapes, degree switched from 2 to 3. # PARAMS: 180k, ACC: ~8%
4. data_shapes = [(3, 32, 32), (32, 32), (32,32), (32,32), (32,32), (512,), (256,), (128,), (100,)], degree = 2 # PARAMS: 120k, ACC: ~4%
5. DATA_SHAPES=[(3, 32, 32), (32, 32)x8, (512,), (256,), (128,), (100,)], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 40k, ACC: 3%
6. DATA_SHAPES=[(3, 32, 32), (64, 64), (32, 32), 1024, 1024, 512, 512, 256, 256, 128, 128, 128, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 40k, ACC: 3%
7. DATA_SHAPES=[(3, 32, 32), (3, 32, 32)*4, 1024, 512, 256, 128, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 80k, ACC: 5% 
8. DATA_SHAPES=[(3, 32, 32), (32, 32, 32)*4, 1024, 512, 256, 128, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 795k, ACC: 5%
9. DATA_SHAPES=[(3, 32, 32), (64, 64), (64, 64), (64, 64), (64, 64), 1024, 512, 256, 128, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 80k, ACC: 5.5%
10. DATA_SHAPES=[(3, 32, 32), 2048, 1024, 512, 256, 128, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 16k, ACC: 4%
11. DATA_SHAPES=[(3, 32, 32), (64, 64), (32, 32), (16, 16), 256, 100], DEGREE=2, BATCH_SIZE=512, DEGREE=2, LR=0.0001 # PARAMS: 28k, ACC: 3.5%
12. SAME AS ABOVE, BUT WITH REGULARLY SPACED INIT. PARAMS: 28k, ACC: 1.5%
```
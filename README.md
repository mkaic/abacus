
# The idea
It boils down to: "what if sparsely connected neurons could slowly move their connection points around?". I achieve this by interpolating each layer's activations at points parameterized by the *next* layer!

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```
```@meta
CurrentModule = NearestNeighborsTest
```

# NearestNeighborsTest.jl

NearestNeighborsTest.jl provides nearest-neighbor distance (NND) permutation tests for 1-D data, with optional CUDA GPU acceleration.

## Exported Functions

```@docs
nnd_permutation_test_1d
nnd_sensitivity_batch_1d
```

## Internals

```@docs
NearestNeighborsTest.mean_knn_within_group_1d
```

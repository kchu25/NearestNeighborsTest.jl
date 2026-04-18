module NearestNeighborsTest

using Random
using Statistics
using CUDA

include("utils.jl")
include("nnd.jl")
include("cuda.jl")

export nnd_permutation_test_1d, nnd_sensitivity_batch_1d

end

"""
    nnd_permutation_test_1d(subpop_positions, background; k=30, B=1_000, seed=42, cuda=false)

Test whether the points at `subpop_positions` within `background` are more
tightly clustered **among themselves** than a random draw of the same size,
using within-group k-NN distances and a permutation test.

**Observed statistic**: sort the `m` subpopulation values → compute the mean
k-th nearest-neighbor distance within that sorted group.

**Null distribution**: draw `m` values at random from `background` → sort →
compute the same within-group mean k-NN distance.  Repeat `B` times.

p-value = fraction of null draws whose within-group mNND ≤ observed.

# Arguments
- `subpop_positions`: Integer vector of positional indices (1-based) into
  `background` identifying the subpopulation members.
- `background`: Full background values (real-valued vector, length N).

# Keyword Arguments
- `k::Int=30`: Number of nearest neighbors.
- `B::Int=1_000`: Number of permutations.
- `seed::Int=42`: Random seed for reproducibility.
- `cuda::Bool=false`: If `true`, use the GPU-accelerated implementation.

# Returns
`NamedTuple` `(k, obs_mNND, p_value)`.
"""
function nnd_permutation_test_1d(
    subpop_positions::AbstractVector{<:Integer},
    background::AbstractVector{T};
    k::Int    = 30,
    B::Int    = 1_000,
    seed::Int = 42,
    cuda::Bool = false,
) where {T<:Real}
    if cuda
        return _nnd_permutation_test_1d_gpu(subpop_positions, background;
                                            k=k, B=B, seed=seed)
    end

    rng    = Random.MersenneTwister(seed)
    n_bg   = length(background)
    n_sub  = length(subpop_positions)
    k_c    = min(k, n_sub - 1)

    # Observed statistic
    subpop_vals = sort!([background[i] for i in subpop_positions])
    obs_mNND    = mean_knn_within_group_1d(subpop_vals, k_c)

    # Null distribution
    count_significant = 0
    perm_indices = Vector{Int}(undef, n_bg)
    null_vals    = Vector{T}(undef, n_sub)

    for _ in 1:B
        Random.randperm!(rng, perm_indices)
        @inbounds for i in 1:n_sub
            null_vals[i] = background[perm_indices[i]]
        end
        sort!(null_vals)
        null_mNND = mean_knn_within_group_1d(null_vals, k_c)
        if null_mNND <= obs_mNND
            count_significant += 1
        end
    end

    return (; k=k_c, obs_mNND, p_value=count_significant / B)
end

"""
    nnd_sensitivity_batch_1d(subpop_positions, background; ks, B=1_000, seed=42, cuda=false)

Batched NND permutation test across multiple `k` values.

Shares the same random draws across all `k` values for efficiency. Each `k`
is tested using **within-group** k-NN distances.

# Arguments
- `subpop_positions`: Integer vector of positional indices (1-based) into
  `background`.
- `background`: Full background values (real-valued vector, length N).

# Keyword Arguments
- `ks::Vector{Int}`: k values to evaluate.
- `B::Int=1_000`: Number of permutations.
- `seed::Int=42`: Random seed.
- `cuda::Bool=false`: If `true`, use the GPU-accelerated implementation.

# Returns
`Vector` of `NamedTuple`s `(k, obs_mNND, p_value)`, one per entry in `ks`.
"""
function nnd_sensitivity_batch_1d(
    subpop_positions::AbstractVector{<:Integer},
    background::AbstractVector{T};
    ks::Vector{Int},
    B::Int    = 1_000,
    seed::Int = 42,
    cuda::Bool = false,
) where {T<:Real}
    if cuda
        return _nnd_sensitivity_batch_1d_gpu(subpop_positions, background;
                                             ks=ks, B=B, seed=seed)
    end

    rng   = Random.MersenneTwister(seed)
    n_bg  = length(background)
    n_sub = length(subpop_positions)

    ks_clamped = [min(kv, n_sub - 1) for kv in ks]
    n_ks       = length(ks)

    # Observed statistics
    subpop_vals = sort!([background[i] for i in subpop_positions])
    obs_mNNDs   = [mean_knn_within_group_1d(subpop_vals, ku) for ku in ks_clamped]

    # Null distribution
    counts_significant = zeros(Int, n_ks)
    perm_indices = Vector{Int}(undef, n_bg)
    null_vals    = Vector{T}(undef, n_sub)

    for _ in 1:B
        Random.randperm!(rng, perm_indices)
        @inbounds for i in 1:n_sub
            null_vals[i] = background[perm_indices[i]]
        end
        sort!(null_vals)
        for (j, ku) in enumerate(ks_clamped)
            null_mNND = mean_knn_within_group_1d(null_vals, ku)
            if null_mNND <= obs_mNNDs[j]
                counts_significant[j] += 1
            end
        end
    end

    inv_B = 1.0 / B
    return [(; k=ks_clamped[j], obs_mNND=obs_mNNDs[j],
              p_value=counts_significant[j] * inv_B) for j in 1:n_ks]
end

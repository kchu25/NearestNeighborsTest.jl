# ─────────────────────────────────────────────────────────────────────
#  CUDA kernel: one block per permutation
#
#  Shared-memory layout (Float32, length = m_padded + nthreads):
#    [1 .. m_padded]              : sort buffer
#    [m_padded+1 .. m_padded+nt] : reduction buffer
#
#  Random sampling is with-replacement via pre-generated U[0,1) floats
#  produced entirely on-device by cuRAND.  Zero CPU loop, zero PCIe
#  transfer of random data.
# ─────────────────────────────────────────────────────────────────────
function _nnd_perm_kernel!(
    counter,        # CuDeviceVector{Int32,1}   length 1
    bg,             # CuDeviceVector{Float32,1}  length n_bg
    rand_floats,    # CuDeviceMatrix{Float32,2}  m × B  (U[0,1))
    n_bg::Int32,
    obs_mNND::Float32,
    m::Int32,
    m_padded::Int32,
    k::Int32,
)
    b   = blockIdx().x
    tid = threadIdx().x
    nt  = Int32(blockDim().x)

    shmem = CuDynamicSharedArray(Float32, m_padded + nt)

    # ── gather: convert U[0,1) → index → background value ──
    i = tid
    while i <= m
        idx = min(unsafe_trunc(Int32, rand_floats[i, b] * n_bg) + Int32(1), n_bg)
        @inbounds shmem[i] = bg[idx]
        i += nt
    end
    i = m + tid
    while i <= m_padded
        @inbounds shmem[i] = Inf32
        i += nt
    end
    sync_threads()

    # ── in-place ascending bitonic sort ──
    k_sort = Int32(2)
    while k_sort <= m_padded
        j = k_sort >> Int32(1)
        while j >= Int32(1)
            t0 = tid - Int32(1)
            while t0 < m_padded
                p0 = t0 ⊻ j
                if p0 > t0
                    a  = t0 + Int32(1)
                    b_ = p0 + Int32(1)
                    ascending = (t0 & k_sort) == Int32(0)
                    @inbounds va = shmem[a]
                    @inbounds vb = shmem[b_]
                    if (ascending & (va > vb)) | (!ascending & (va < vb))
                        @inbounds shmem[a]  = vb
                        @inbounds shmem[b_] = va
                    end
                end
                t0 += nt
            end
            sync_threads()
            j >>= Int32(1)
        end
        k_sort <<= Int32(1)
    end

    # ── k-th NN distance per point ──
    local_sum = Float32(0)
    i = tid
    while i <= m
        li = i - Int32(1)
        ri = i + Int32(1)
        knn_d = Float32(0)
        for _ in Int32(1):k
            dl = li >= Int32(1) ? @inbounds(shmem[i]  - shmem[li]) : Inf32
            dr = ri <= m        ? @inbounds(shmem[ri] - shmem[i])  : Inf32
            if dl <= dr
                knn_d = dl; li -= Int32(1)
            else
                knn_d = dr; ri += Int32(1)
            end
        end
        local_sum += knn_d
        i += nt
    end

    # ── parallel tree reduction ──
    @inbounds shmem[m_padded + tid] = local_sum
    sync_threads()
    s = nt >> Int32(1)
    while s > Int32(0)
        if tid <= s
            @inbounds shmem[m_padded + tid] += shmem[m_padded + tid + s]
        end
        sync_threads()
        s >>= Int32(1)
    end

    # ── compare and accumulate ──
    if tid == Int32(1)
        null_mNND = @inbounds shmem[m_padded + Int32(1)] / Float32(m)
        if null_mNND <= obs_mNND
            CUDA.@atomic counter[1] += Int32(1)
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  CUDA kernel: sensitivity batch – loops over k values
# ─────────────────────────────────────────────────────────────────────
function _nnd_sensitivity_kernel!(
    counters,       # CuDeviceVector{Int32,1}   length n_ks
    bg,             # CuDeviceVector{Float32,1}
    rand_floats,    # CuDeviceMatrix{Float32,2}  m × B
    obs_mNNDs,      # CuDeviceVector{Float32,1}  length n_ks
    ks_dev,         # CuDeviceVector{Int32,1}    length n_ks
    n_ks::Int32,
    n_bg::Int32,
    m::Int32,
    m_padded::Int32,
)
    b   = blockIdx().x
    tid = threadIdx().x
    nt  = Int32(blockDim().x)

    shmem = CuDynamicSharedArray(Float32, m_padded + nt)

    # ── gather + pad ──
    i = tid
    while i <= m
        idx = min(unsafe_trunc(Int32, rand_floats[i, b] * n_bg) + Int32(1), n_bg)
        @inbounds shmem[i] = bg[idx]
        i += nt
    end
    i = m + tid
    while i <= m_padded
        @inbounds shmem[i] = Inf32
        i += nt
    end
    sync_threads()

    # ── bitonic sort ──
    k_sort = Int32(2)
    while k_sort <= m_padded
        j = k_sort >> Int32(1)
        while j >= Int32(1)
            t0 = tid - Int32(1)
            while t0 < m_padded
                p0 = t0 ⊻ j
                if p0 > t0
                    a  = t0 + Int32(1)
                    b_ = p0 + Int32(1)
                    ascending = (t0 & k_sort) == Int32(0)
                    @inbounds va = shmem[a]
                    @inbounds vb = shmem[b_]
                    if (ascending & (va > vb)) | (!ascending & (va < vb))
                        @inbounds shmem[a]  = vb
                        @inbounds shmem[b_] = va
                    end
                end
                t0 += nt
            end
            sync_threads()
            j >>= Int32(1)
        end
        k_sort <<= Int32(1)
    end

    # ── evaluate each k, reusing sorted buffer ──
    for ki in Int32(1):n_ks
        kv = @inbounds ks_dev[ki]

        local_sum = Float32(0)
        i = tid
        while i <= m
            li = i - Int32(1)
            ri = i + Int32(1)
            knn_d = Float32(0)
            for _ in Int32(1):kv
                dl = li >= Int32(1) ? @inbounds(shmem[i]  - shmem[li]) : Inf32
                dr = ri <= m        ? @inbounds(shmem[ri] - shmem[i])  : Inf32
                if dl <= dr
                    knn_d = dl; li -= Int32(1)
                else
                    knn_d = dr; ri += Int32(1)
                end
            end
            local_sum += knn_d
            i += nt
        end

        @inbounds shmem[m_padded + tid] = local_sum
        sync_threads()
        s = nt >> Int32(1)
        while s > Int32(0)
            if tid <= s
                @inbounds shmem[m_padded + tid] += shmem[m_padded + tid + s]
            end
            sync_threads()
            s >>= Int32(1)
        end

        if tid == Int32(1)
            null_mNND = @inbounds shmem[m_padded + Int32(1)] / Float32(m)
            if null_mNND <= @inbounds obs_mNNDs[ki]
                CUDA.@atomic counters[ki] += Int32(1)
            end
        end
        sync_threads()
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────
#  Host wrappers
# ─────────────────────────────────────────────────────────────────────

"""
    _nnd_permutation_test_1d_gpu(subpop_positions, background; k, B, seed)

GPU-accelerated single-k NND permutation test (internal).

All `B×m` random samples are generated on-device via cuRAND — no CPU
randperm loop, no PCIe transfer of random indices.  One CUDA block per
permutation with shared-memory bitonic sort + k-NN scan.
"""
function _nnd_permutation_test_1d_gpu(
    subpop_positions::AbstractVector{<:Integer},
    background::AbstractVector{T};
    k::Int    = 30,
    B::Int    = 1_000,
    seed::Int = 42,
) where {T<:Real}
    n_bg = length(background)
    m    = length(subpop_positions)
    k_c  = min(k, m - 1)

    subpop_vals = sort!(Float64[background[i] for i in subpop_positions])
    obs         = Float32(mean_knn_within_group_1d(subpop_vals, k_c))

    bg_dev  = CuVector{Float32}(background)
    rf_dev  = CUDA.rand(Float32, m, B)
    CUDA.synchronize()
    cnt_dev = CUDA.zeros(Int32, 1)

    m_pad    = Int32(_next_pow2(m))
    nthreads = Int32(min(1024, Int(m_pad)))
    shmem    = Int((m_pad + nthreads) * sizeof(Float32))

    @cuda(
        blocks  = B,
        threads = nthreads,
        shmem   = shmem,
        _nnd_perm_kernel!(cnt_dev, bg_dev, rf_dev,
                          Int32(n_bg), obs,
                          Int32(m), m_pad, Int32(k_c))
    )
    CUDA.synchronize()
    p_value = Int(Array(cnt_dev)[1]) / B
    return (; k=k_c, obs_mNND=Float64(obs), p_value)
end

"""
    _nnd_sensitivity_batch_1d_gpu(subpop_positions, background; ks, B, seed)

GPU-accelerated sensitivity batch (internal). One sort per permutation
block; k-NN evaluated for each k reusing the already-sorted shared-memory
buffer.
"""
function _nnd_sensitivity_batch_1d_gpu(
    subpop_positions::AbstractVector{<:Integer},
    background::AbstractVector{T};
    ks::Vector{Int},
    B::Int    = 1_000,
    seed::Int = 42,
) where {T<:Real}
    n_bg = length(background)
    m    = length(subpop_positions)
    n_ks = length(ks)

    ks_clamped  = [min(kv, m - 1) for kv in ks]
    subpop_vals = sort!(Float64[background[i] for i in subpop_positions])
    obs_f64     = [mean_knn_within_group_1d(subpop_vals, ku) for ku in ks_clamped]

    bg_dev  = CuVector{Float32}(background)
    rf_dev  = CUDA.rand(Float32, m, B)
    CUDA.synchronize()
    cnt_dev = CUDA.zeros(Int32, n_ks)
    obs_dev = CuVector{Float32}(Float32.(obs_f64))
    ks_dev  = CuVector{Int32}(Int32.(ks_clamped))

    m_pad    = Int32(_next_pow2(m))
    nthreads = Int32(min(1024, Int(m_pad)))
    shmem    = Int((m_pad + nthreads) * sizeof(Float32))

    @cuda(
        blocks  = B,
        threads = nthreads,
        shmem   = shmem,
        _nnd_sensitivity_kernel!(cnt_dev, bg_dev, rf_dev,
                                 obs_dev, ks_dev, Int32(n_ks),
                                 Int32(n_bg), Int32(m), m_pad)
    )
    CUDA.synchronize()
    counts = Array(cnt_dev)
    inv_B  = 1.0 / B
    return [(; k=ks_clamped[j], obs_mNND=obs_f64[j],
              p_value=counts[j] * inv_B) for j in 1:n_ks]
end

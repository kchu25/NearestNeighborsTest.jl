"""
    _next_pow2(n::Int) -> Int

Smallest power of two ≥ `n`.
"""
_next_pow2(n::Int) = 1 << ceil(Int, log2(max(n, 1)))

"""
    mean_knn_within_group_1d(sorted_vals, k)

Compute the mean k-NN distance **within** a sorted group of 1-D values.

For each point, scan left and right among its group-mates to find the `k`
nearest neighbors, then average the k-th NN distance across all points.
Since the group is already sorted, neighbors are always adjacent — O(m·k).
"""
function mean_knn_within_group_1d(sorted_vals::AbstractVector{T}, k::Int) where {T<:Real}
    m = length(sorted_vals)
    k_clamped = min(k, m - 1)
    sum_knn_dists = 0.0
    for i in 1:m
        left_idx, right_idx = i - 1, i + 1
        knn_dist = 0.0
        for _ in 1:k_clamped
            dist_left  = left_idx  >= 1 ? sorted_vals[i] - sorted_vals[left_idx]  : Inf
            dist_right = right_idx <= m ? sorted_vals[right_idx] - sorted_vals[i]  : Inf
            if dist_left <= dist_right
                knn_dist = dist_left
                left_idx -= 1
            else
                knn_dist = dist_right
                right_idx += 1
            end
        end
        sum_knn_dists += knn_dist
    end
    return sum_knn_dists / m
end

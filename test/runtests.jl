using NearestNeighborsTest
using Random
using Test

# Convenience alias for the internal helper
const mKNN = NearestNeighborsTest.mean_knn_within_group_1d
const pow2  = NearestNeighborsTest._next_pow2

@testset "NearestNeighborsTest.jl" begin

    # ─────────────────────────────────────────────────────────────────
    @testset "_next_pow2" begin
        @test pow2(1)  == 1
        @test pow2(2)  == 2
        @test pow2(3)  == 4
        @test pow2(4)  == 4
        @test pow2(5)  == 8
        @test pow2(7)  == 8
        @test pow2(8)  == 8
        @test pow2(9)  == 16
        @test pow2(100) == 128
        @test pow2(1024) == 1024
        @test pow2(1025) == 2048
    end

    # ─────────────────────────────────────────────────────────────────
    @testset "mean_knn_within_group_1d" begin

        @testset "evenly spaced, k=1" begin
            vals = collect(1.0:5.0)          # spacing = 1
            @test mKNN(vals, 1) ≈ 1.0
        end

        @testset "evenly spaced, k=2" begin
            # k=2 NN distances for [1,2,3,4,5] (spacing=1):
            # point 1: 2nd-NN is 3 → d=2
            # point 2: goes left to 1 (d=1), then right to 3 (d=1) → knn_d=1
            # point 3: goes left to 2 (d=1), then right to 4 (d=1) → knn_d=1
            # point 4: goes left to 3 (d=1), then right to 5 (d=1) → knn_d=1
            # point 5: 2nd-NN is 3 → d=2
            # mean = (2+1+1+1+2)/5 = 1.4
            vals = collect(1.0:5.0)
            @test mKNN(vals, 2) ≈ 1.4
        end

        @testset "single point – k clamped to 0" begin
            @test mKNN([3.0], 1)  == 0.0
            @test mKNN([3.0], 99) == 0.0
        end

        @testset "two points – k=1" begin
            # mean 1-NN distance = distance between the two = 5
            @test mKNN([0.0, 5.0], 1) ≈ 5.0
        end

        @testset "k clamped to m-1" begin
            vals = collect(1.0:4.0)   # m=4, k_clamped = 3
            d_large = mKNN(vals, 100)
            d_exact = mKNN(vals, 3)
            @test d_large ≈ d_exact
        end

        @testset "left boundary (Inf on left side)" begin
            # Leftmost point must expand right only
            vals = [0.0, 1.0, 2.0, 10.0, 11.0]
            d = mKNN(vals, 1)
            @test d > 0.0
        end

        @testset "right boundary (Inf on right side)" begin
            # Rightmost point must expand left only
            vals = [0.0, 1.0, 9.0, 10.0, 11.0]
            d = mKNN(vals, 1)
            @test d > 0.0
        end

        @testset "Float32 input" begin
            vals = Float32[1, 2, 3, 4, 5]
            @test mKNN(vals, 1) ≈ 1.0
        end

        @testset "all identical values" begin
            vals = [0.5, 0.5, 0.5, 0.5]
            @test mKNN(vals, 1) == 0.0
            @test mKNN(vals, 3) == 0.0
        end

        @testset "known analytic result" begin
            # [0, 1, 3, 6] with k=1:
            # point 0: right=1, d=1
            # point 1: left=1,right=2 → d=1
            # point 3: left=2,right=3 → d=2
            # point 6: left=3, d=3
            # mean = (1+1+2+3)/4 = 1.75
            vals = [0.0, 1.0, 3.0, 6.0]
            @test mKNN(vals, 1) ≈ 1.75
        end
    end

    # ─────────────────────────────────────────────────────────────────
    @testset "nnd_permutation_test_1d (CPU)" begin

        @testset "clustered → small p-value" begin
            rng = MersenneTwister(1)
            n_bg = 10_000;  m = 200
            bg = rand(rng, Float64, n_bg)
            pos = sort(randperm(rng, n_bg)[1:m])
            for p in pos; bg[p] = 0.5 + 0.01 * randn(rng); end

            r = nnd_permutation_test_1d(pos, bg; k=10, B=500, seed=42)
            @test r.p_value < 0.05
            @test r.k == 10
            @test r.obs_mNND > 0.0
        end

        @testset "null (random subpop) → p not tiny" begin
            rng = MersenneTwister(99)
            n_bg = 5_000;  m = 100
            bg  = rand(rng, Float64, n_bg)
            pos = sort(randperm(rng, n_bg)[1:m])

            r = nnd_permutation_test_1d(pos, bg; k=5, B=500, seed=42)
            @test r.p_value > 0.01
        end

        @testset "k clamped when k ≥ n_subpop" begin
            rng = MersenneTwister(7)
            bg  = rand(rng, 1_000)
            pos = sort(randperm(rng, 1_000)[1:10])

            r = nnd_permutation_test_1d(pos, bg; k=50, B=100, seed=1)
            @test r.k == 9   # clamped to m-1
            @test 0.0 <= r.p_value <= 1.0
        end

        @testset "reproducibility – same seed → same result" begin
            rng = MersenneTwister(3)
            bg  = rand(rng, 2_000)
            pos = sort(randperm(rng, 2_000)[1:50])

            r1 = nnd_permutation_test_1d(pos, bg; k=5, B=200, seed=77)
            r2 = nnd_permutation_test_1d(pos, bg; k=5, B=200, seed=77)
            @test r1.p_value   == r2.p_value
            @test r1.obs_mNND  == r2.obs_mNND
            @test r1.k         == r2.k
        end

        @testset "different seeds → different p-values" begin
            rng = MersenneTwister(5)
            bg  = rand(rng, 2_000)
            pos = sort(randperm(rng, 2_000)[1:50])

            r1 = nnd_permutation_test_1d(pos, bg; k=5, B=200, seed=1)
            r2 = nnd_permutation_test_1d(pos, bg; k=5, B=200, seed=2)
            # obs_mNND is deterministic (no random); p-values may differ
            @test r1.obs_mNND == r2.obs_mNND
            # p-values will almost certainly differ (probabilistic, but reliable)
            # (we don't assert inequality to avoid flakiness)
        end

        @testset "Float32 background" begin
            rng = MersenneTwister(11)
            bg  = rand(rng, Float32, 2_000)
            pos = sort(randperm(rng, 2_000)[1:40])
            for p in pos; bg[p] = 0.5f0 + 0.01f0 * randn(rng, Float32); end

            r = nnd_permutation_test_1d(pos, bg; k=5, B=200, seed=1)
            @test 0.0 <= r.p_value <= 1.0
            @test r.obs_mNND > 0.0
        end

        @testset "B=1 (minimal permutations)" begin
            rng = MersenneTwister(13)
            bg  = rand(rng, 500)
            pos = sort(randperm(rng, 500)[1:20])

            r = nnd_permutation_test_1d(pos, bg; k=3, B=1, seed=1)
            @test r.p_value in (0.0, 1.0)
        end

        @testset "perfectly clustered → p_value = 0" begin
            # All subpop values at exactly 0.5 — far tighter than any random draw
            rng = MersenneTwister(17)
            bg  = rand(rng, 5_000)
            pos = 1:50
            for p in pos; bg[p] = 0.5; end

            r = nnd_permutation_test_1d(collect(pos), bg; k=5, B=500, seed=42)
            @test r.p_value == 0.0
            @test r.obs_mNND == 0.0
        end

        @testset "p_value always in [0, 1]" begin
            for seed in 1:5
                rng = MersenneTwister(seed)
                bg  = rand(rng, 2_000)
                pos = sort(randperm(rng, 2_000)[1:30])
                r   = nnd_permutation_test_1d(pos, bg; k=5, B=100, seed=seed)
                @test 0.0 <= r.p_value <= 1.0
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    @testset "nnd_sensitivity_batch_1d (CPU)" begin

        @testset "basic structure and p-value bounds" begin
            rng = MersenneTwister(1)
            n_bg = 10_000;  m = 200
            bg  = rand(rng, Float64, n_bg)
            pos = sort(randperm(rng, n_bg)[1:m])
            for p in pos; bg[p] = 0.5 + 0.01 * randn(rng); end

            ks = [5, 10, 20]
            results = nnd_sensitivity_batch_1d(pos, bg; ks=ks, B=500, seed=42)
            @test length(results) == 3
            for r in results
                @test haskey(r, :k)
                @test haskey(r, :obs_mNND)
                @test haskey(r, :p_value)
                @test 0.0 <= r.p_value <= 1.0
                @test r.obs_mNND > 0.0
            end
        end

        @testset "clustered → all p-values small" begin
            rng = MersenneTwister(1)
            n_bg = 10_000;  m = 200
            bg  = rand(rng, Float64, n_bg)
            pos = sort(randperm(rng, n_bg)[1:m])
            for p in pos; bg[p] = 0.5 + 0.01 * randn(rng); end

            results = nnd_sensitivity_batch_1d(pos, bg; ks=[5, 10, 20], B=500, seed=42)
            for r in results
                @test r.p_value < 0.1
            end
        end

        @testset "single k" begin
            rng = MersenneTwister(2)
            bg  = rand(rng, 3_000)
            pos = sort(randperm(rng, 3_000)[1:60])

            results = nnd_sensitivity_batch_1d(pos, bg; ks=[10], B=200, seed=1)
            @test length(results) == 1
            @test 0.0 <= results[1].p_value <= 1.0
        end

        @testset "k values clamped to m-1" begin
            rng = MersenneTwister(4)
            bg  = rand(rng, 1_000)
            pos = sort(randperm(rng, 1_000)[1:10])   # m=10

            results = nnd_sensitivity_batch_1d(pos, bg; ks=[5, 50, 200], B=100, seed=1)
            # all k > 9 should be clamped to 9
            @test results[2].k == 9
            @test results[3].k == 9
        end

        @testset "consistency with single-test for same k and seed" begin
            rng = MersenneTwister(6)
            bg  = rand(rng, 3_000)
            pos = sort(randperm(rng, 3_000)[1:60])

            r_single = nnd_permutation_test_1d(pos, bg; k=10, B=300, seed=55)
            r_batch  = nnd_sensitivity_batch_1d(pos, bg; ks=[10], B=300, seed=55)

            # obs_mNND must be exactly equal (same CPU computation)
            @test r_single.obs_mNND == r_batch[1].obs_mNND
            @test r_single.k        == r_batch[1].k
            # p-values should be equal (same RNG path for a single k)
            @test r_single.p_value  ≈ r_batch[1].p_value  atol=1e-12
        end

        @testset "reproducibility" begin
            rng = MersenneTwister(8)
            bg  = rand(rng, 2_000)
            pos = sort(randperm(rng, 2_000)[1:50])

            r1 = nnd_sensitivity_batch_1d(pos, bg; ks=[5, 15], B=200, seed=99)
            r2 = nnd_sensitivity_batch_1d(pos, bg; ks=[5, 15], B=200, seed=99)
            for (a, b) in zip(r1, r2)
                @test a.p_value  == b.p_value
                @test a.obs_mNND == b.obs_mNND
            end
        end

        @testset "Float32 background" begin
            rng = MersenneTwister(12)
            bg  = rand(rng, Float32, 2_000)
            pos = sort(randperm(rng, 2_000)[1:40])
            for p in pos; bg[p] = 0.5f0 + 0.01f0 * randn(rng, Float32); end

            results = nnd_sensitivity_batch_1d(pos, bg; ks=[5, 10], B=100, seed=1)
            @test length(results) == 2
            for r in results
                @test 0.0 <= r.p_value <= 1.0
            end
        end

        @testset "obs_mNND non-decreasing with k (sorted group)" begin
            # Within a clustered group, larger k means larger distance
            rng = MersenneTwister(20)
            n_bg = 5_000;  m = 100
            bg  = rand(rng, n_bg)
            pos = sort(randperm(rng, n_bg)[1:m])
            for p in pos; bg[p] = 0.5 + 0.01 * randn(rng); end

            ks = [1, 5, 10, 20]
            results = nnd_sensitivity_batch_1d(pos, bg; ks=ks, B=100, seed=1)
            obs = [r.obs_mNND for r in results]
            @test issorted(obs)
        end
    end

end


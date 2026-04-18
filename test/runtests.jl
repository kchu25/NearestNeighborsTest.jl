using NearestNeighborsTest
using Random
using Test

@testset "NearestNeighborsTest.jl" begin

    @testset "mean_knn_within_group_1d" begin
        # Evenly spaced points: k=1 NN distance should be the spacing
        vals = collect(1.0:5.0)
        d = NearestNeighborsTest.mean_knn_within_group_1d(vals, 1)
        @test d ≈ 1.0

        # Single point: k gets clamped to 0, mean distance = 0
        d_single = NearestNeighborsTest.mean_knn_within_group_1d([3.0], 5)
        @test d_single == 0.0
    end

    @testset "nnd_permutation_test_1d (CPU)" begin
        # Clustered subpopulation → should yield small p-value
        rng = MersenneTwister(1)
        n_bg = 10_000
        m = 200
        bg = rand(rng, Float64, n_bg)
        pos = sort(randperm(rng, n_bg)[1:m])
        for p in pos
            bg[p] = 0.5 + 0.01 * randn(rng)
        end

        r = nnd_permutation_test_1d(pos, bg; k=10, B=500, seed=42, cuda=false)
        @test r.p_value < 0.05
        @test r.k == 10
        @test r.obs_mNND > 0.0
    end

    @testset "nnd_permutation_test_1d null calibration" begin
        # Random subpopulation → p-value should NOT be tiny
        rng = MersenneTwister(99)
        n_bg = 5_000
        m = 100
        bg = rand(rng, Float64, n_bg)
        pos = sort(randperm(rng, n_bg)[1:m])

        r = nnd_permutation_test_1d(pos, bg; k=5, B=500, seed=42, cuda=false)
        @test r.p_value > 0.01  # should not be significant under null
    end

    @testset "nnd_sensitivity_batch_1d (CPU)" begin
        rng = MersenneTwister(1)
        n_bg = 10_000
        m = 200
        bg = rand(rng, Float64, n_bg)
        pos = sort(randperm(rng, n_bg)[1:m])
        for p in pos
            bg[p] = 0.5 + 0.01 * randn(rng)
        end

        ks = [5, 10, 20]
        results = nnd_sensitivity_batch_1d(pos, bg; ks=ks, B=500, seed=42, cuda=false)
        @test length(results) == 3
        for r in results
            @test haskey(r, :k)
            @test haskey(r, :obs_mNND)
            @test haskey(r, :p_value)
            @test 0.0 <= r.p_value <= 1.0
        end
    end

end

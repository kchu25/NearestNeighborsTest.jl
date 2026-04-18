using CairoMakie
using Random

# ─────────────────────────────────────────────────────────────────────
#  Illustration 1: Clustered vs Random Subpopulation
# ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(1400, 400))

# Clustered subpopulation (left)
ax1 = Axis(fig[1, 1];
    limits=((-0.1, 1.1), (-1, 1)),
    xlabel="Value", ylabel="",
    title="Clustered Subpopulation\n(significant, p < 0.05)",
    yticklabelsvisible=false)

rng = MersenneTwister(1)
n_bg = 100
bg_vals = rand(rng, n_bg)
n_sub = 25
sub_pos = sort(randperm(rng, n_bg)[1:n_sub])
sub_vals = similar(bg_vals)
for p in sub_pos
    bg_vals[p] = 0.5 + 0.05 * randn(rng)
end

# Plot background (gray dots)
bg_other = filter(i -> !(i in sub_pos), 1:n_bg)
scatter!(ax1, bg_vals[bg_other], zeros(length(bg_other)); 
    color=:lightgray, markersize=8, label="background")
# Plot subpopulation (red dots)
scatter!(ax1, bg_vals[sub_pos], zeros(length(sub_pos)); 
    color=:red, markersize=10, label="subpopulation")
axislegend(ax1; position=:rb)

# Random subpopulation (right)
ax2 = Axis(fig[1, 2];
    limits=((-0.1, 1.1), (-1, 1)),
    xlabel="Value", ylabel="",
    title="Random Subpopulation\n(not significant, p ≈ 0.5)",
    yticklabelsvisible=false)

rng2 = MersenneTwister(99)
n_bg2 = 100
bg_vals2 = rand(rng2, n_bg2)
n_sub2 = 25
sub_pos2 = sort(randperm(rng2, n_bg2)[1:n_sub2])

bg_other2 = filter(i -> !(i in sub_pos2), 1:n_bg2)
scatter!(ax2, bg_vals2[bg_other2], zeros(length(bg_other2)); 
    color=:lightgray, markersize=8, label="background")
scatter!(ax2, bg_vals2[sub_pos2], zeros(length(sub_pos2)); 
    color=:blue, markersize=10, label="subpopulation")
axislegend(ax2; position=:rb)

save("assets/illustration_concept.png", fig; px_per_unit=2)
println("Saved: assets/illustration_concept.png")

# ─────────────────────────────────────────────────────────────────────
#  Illustration 2: k-NN Distance Concept
# ─────────────────────────────────────────────────────────────────────
fig2 = Figure(size=(1200, 350))

ax3 = Axis(fig2[1, 1];
    limits=((0, 1), (-0.5, 2.5)),
    xlabel="Value", ylabel="",
    title="Within-Group k-NN Distances (k=3)",
    yticklabelsvisible=false)

# Sorted subpopulation with spacing annotations
rng3 = MersenneTwister(42)
sorted_sub = sort(rand(rng3, 10))

# Plot sorted points
scatter!(ax3, sorted_sub, zeros(length(sorted_sub)); 
    color=:darkblue, markersize=12)

# Mark a few points and show their k=3 nearest neighbors
highlight_idx = [3, 5, 8]
colors_hl = [:red, :green, :orange]

for (hi, col) in zip(highlight_idx, colors_hl)
    x = sorted_sub[hi]
    scatter!(ax3, [x], [0]; color=col, markersize=14)
    
    # Find 3 nearest neighbors
    dists = [abs(sorted_sub[j] - x) for j in 1:length(sorted_sub) if j != hi]
    sorted_dists = sort(dists)[1:3]
    max_dist = sorted_dists[end]
    
    # Draw arc showing k=3 distance
    θ = range(0, π, length=50)
    arc_x = x .+ max_dist/2 .* cos.(θ)
    arc_y = 0.3 .+ max_dist/2 .* sin.(θ)
    lines!(ax3, arc_x, arc_y; color=col, linewidth=2, alpha=0.7)
end

text!(ax3, 0.05, 2.0; text="Red/Green/Orange arcs\nshow k=3 NN distance\nfor each highlighted point", 
    fontsize=10, align=(:left, :center))

save("assets/illustration_knn.png", fig2; px_per_unit=2)
println("Saved: assets/illustration_knn.png")

using NearestNeighborsTest
using Documenter

DocMeta.setdocmeta!(NearestNeighborsTest, :DocTestSetup, :(using NearestNeighborsTest); recursive=true)

makedocs(;
    modules=[NearestNeighborsTest],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="NearestNeighborsTest.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/NearestNeighborsTest.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/NearestNeighborsTest.jl",
    devbranch="main",
)

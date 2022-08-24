using GLM, GLMNet
using CSV, DataFrames
x = CSV.read("jl/Data/x_rnd.csv", DataFrame)
y = CSV.read("jl/Data/y_rnd.csv", DataFrame)

x = x[:, 2:21]
y = y[:, 2]


# params

index = 1:size(x, 1)
method = "partial out"
I3 = nothing
post = true

# if method Not

if method ∉ ["Partial out", "double selection"]
    error("invalid method")
end

Vector{Bool}([true, false])

[true, false]
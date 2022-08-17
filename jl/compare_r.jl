using GLM, Distributions, CSV, Tables, DataFrames
using GLMNet

x_rnd = CSV.read("jl/Data/x_rnd.csv", DataFrame)# |> DataFrame()
y_rnd = CSV.read("jl/Data/y_rnd.csv", DataFrame)

x = Array(x_rnd)
y = Array(y_rnd)

x = convert(Matrix, x)
y = y_rnd[!, :x]

glmnet(x_rnd, y_rnd)

glmnet(x, y, )
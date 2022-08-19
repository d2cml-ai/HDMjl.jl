using GLM, GLMNet
using CSV, DataFrames
using Distributions

include("hdmjl.jl")

xt = Matrix(CSV.read("jl/Data/xt_r.csv", DataFrame))
yt = Matrix(CSV.read("jl/Data/yt_r.csv", DataFrame))
dt = Matrix(CSV.read("jl/Data/dt_r.csv", DataFrame))
zt = Matrix(CSV.read("jl/Data/zt_r.csv", DataFrame))


##-- params

x = xt
y = yt
d = dt
z = zt

# x, y, d, z

bootstrap = "none"
n_rep = 50
post = true
intercept = true
always_takers = true

n, p = size(xt)

if bootstrap ∉ ["none", "Bayes", "normal", "wild"] 
    error("Must be element of set [none, Bayes, normal, wild], but bootstrap is $bootstrap")
end

lambda = 2.2 * sqrt(n) * quantile.(Normal(0.0, 1.0), (.1 / log(n)) / (2 * (2 * p)))
## contorl
control_inter = 15
control_tol = 10^-5
# penalty
penalty_homoscedastic = "none"
penalty_lambda_start = repeat([lambda])
penalty_c = 1.1
penalty_gamma = .1


indz1, indz0 = [], []

for i in eachindex(z)
    if z[i] == 1
        append!(indz1, i)
    else
        append!(indz0, i)
    end
end


rlasso(
    x[indz0, :], y[indz0, :], post = post, intercept = intercept
)

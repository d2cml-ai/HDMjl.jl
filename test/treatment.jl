print("PKg")
using Pkg
Pkg.rm("HDMjl")
Pkg.add(url = "https://github.com/d2cml-ai/HDMjl.jl", rev = "test_formula")

using HDMjl

print("data")

pension = get_data("pension")
y = pension[:, "tw"];
d = pension[:, "p401"];
z = pension[:, "e401"];
X = pension[:, ["i2", "i3", "i4", "i5", "i6", "i7", "a2", "a3", "a4", "a5", "fsize", "hs", "smcol", "col", "marr", "twoearn", "db", "pira", "hown"]]


rlassoLATET(X, d, y, z)

indz0 = findall(z .== 0)


using Distributions
n, p = size(X)
lambda = 2.2 * sqrt(n) * quantile(Normal(0.0, 1.0),  1 - (.1 / log(n)) / (2 * (2 * p)))

hcat(ones(size(X, 1)), Matrix(X)) * b_y_z0xL["coefficients"]





length(b_y_z0xL["coefficients"])
include("../src/HDMjl.jl")

x = Matrix(X[:, :])
d = Matrix(d[:, :])
y = Matrix(y[:, :])
z = Matrix(z[:, :])
n = size(x, 1)
p = size(x, 2)

post = true
intercept = true
bootstrap = "none"
n_rep = 500
always_takers = true
never_takers = true


lambda = 2.2 * sqrt(n) * quantile(Normal(0.0, 1.0),  1 - (0.1 / log(n)) / (2 * (2 * p)))
indz0 = findall(z .== 0)
lambda_str = repeat([lambda], p)

indz1, indz0 = [], []

for i in eachindex(z)
    if z[i] == 1
        append!(indz1, i)
    else
        append!(indz0, i)
    end
end

include("../src/HDMjl.jl")
# b_y_z0xL = HDMjl.rlasso(x[indz0, :], y[indz0], post = true, intercept = true,  homoskedastic = "none", gamma = 0.1)
# b_y_z0xL = HDMjl.rlasso(x[indz0, :], y[indz0], post = post, intercept = intercept, homoskedastic = "none", c = 1.1, gamma = 0.1, lambda_start = lambda_str)

indz0

b_y_z0xL["coefficients"]

X


using Pkg
Pkg.rm("HDMjl")
Pkg.add(url = "https://github.com/d2cml-ai/HDMjl.jl", rev = "test_formula")

using HDMjl
pension = get_data("pension")
y = pension[:, "tw"];
d = pension[:, "p401"];
z = pension[:, "e401"];
X = pension[:, ["i2", "i3", "i4", "i5", "i6", "i7", "a2", "a3", "a4", "a5", "fsize", "hs", "smcol", "col", "marr", "twoearn", "db", "pira", "hown"]]
# x = Matrix(X[:, :])
d = Matrix(d[:, :])
y = Matrix(y[:, :])
z = Matrix(z[:, :])
n = size(x, 1)
p = size(x, 2)
HDMjl.r_summary(HDMjl.rlassoLATET(X, d, y, z))
HDMjl.r_summary(HDMjl.rlassoATET(X, d, y))
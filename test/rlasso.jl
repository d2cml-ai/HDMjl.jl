# set.seed(12345)
using  Random, Distributions
n = 100 #sample size
p = 100 # number of variables
s = 3 # nubmer of variables with non-zero coefficients
x = rand(Normal(), (n, p))
# beta = c(rep(5, s), repeat(0, p - s))
beta = vcat(fill(3, s), zeros(p - s));
y =1 .+ x * beta + randn(n);
y
# Y = X %*% beta + rnorm(n)


beta = vcat(repeat([5], s), repeat([0], p - s))

using DataFrames
x1 = DataFrame(x, :auto)


# ] activate .

using Random, Distributions

n = 250
p = 40
px = 10



using DataFrames
x = rand(Normal(), (n, p))

x1 = DataFrame(x, :auto)

y = rand(Binomial(), n)

y1 = DataFrame(y[:, :], :auto)

d = rand(n)

include("../src/HDMjl.jl")

# ss = HDMjl.rlassologitEffect(x1, y, d);
# typeof(ss)

# ss = HDMjl.rlassologitEffect(x, y, d);
ss = HDMjl.rlassologitEffects(x1, y, index = 1:3);


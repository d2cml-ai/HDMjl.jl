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
aa = HDMjl.rlassologit(x1, y);
HDMjl.r_summary(aa)
ss = HDMjl.rlassologitEffect(x, y, d);
# ss = HDMjl.rlassologitEffects(x1, y, index = 1:3);
# HDMjl.r_summary(ss)
# typeof(ss)
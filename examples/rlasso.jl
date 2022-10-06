
using CSV, DataFrames
function r_data(n = 1)
    n_m = "examples/r_" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

reload() = include("../src/HDMjl.jl")

dta = r_data(1)
n, p = size(dta)
s = 3
beta = vcat(fill(3, s), zeros(p - s));

p = p-1
X = dta[:, Not(1)]
Y = dta[:, 1];

# Y

reload()
lasso_reg = HDMjl.rlasso(X, Y, post = false)
HDMjl.r_summary(lasso_reg)

#------
using Pkg, Pkg.rm("HDMjl"); Pkg.add("https://github.com/d2cml-ai/HDMjl.jl#printing")
using  DataFrames, CSV, HDMjl

url = "https://raw.githubusercontent.com/d2cml-ai/HDMjl.jl/printing/examples/r_1.csv"
dta = CSV.read(download(url), DataFrame)
n, p = size(dta)
s = 3
beta = vcat(fill(3, s), zeros(p - s));

p = p-1
X = Matrix(dta[:, Not(1)])
Y = dta[:, 1];
lasso_reg = rlasso(X, Y, post = false)
# HDMjl.r_summary(lasso_reg)

e1

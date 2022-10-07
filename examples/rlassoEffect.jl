using CSV, DataFrames
function r_data(n = 1)
    n_m = "examples/r_" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

reload() = include("../src/HDMjl.jl")

dta = r_data("1_eff")

y = dta[:, 1]
d = dta[:, 2]
x = dta[:, Not(1, 2)]

# reload()


eff = HDMjl.rlassoEffect(Matrix(x), y, d, method = "partialling out")
HDMjl.r_summary(eff); ##3 resultados coinciden 
eff = HDMjl.rlassoEffect(Matrix(x), y, d, method = "double selection")
HDMjl.r_summary(eff); ##3 resultados coinciden 

###
dta_growth = r_data("growth_data")

y = dta_growth[:, 1]
d = dta_growth[:, 2]
x = dta_growth[:, Not(1, 2)]

lasso_effect = HDMjl.rlassoEffect(Matrix(x), y, d, method = "partialling out")
HDMjl.r_summary(lasso_effect);

double_effect = HDMjl.rlassoEffect(Matrix(x), y, d, method = "double selection")
HDMjl.r_summary(double_effect);


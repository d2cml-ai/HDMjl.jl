using CSV, DataFrames
function r_data(n = 1)
    n_m = "examples/r_" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

reload() = include("../src/HDMjl.jl")

effts = r_data("01Effects")
y = effts[:, 1]
x = Matrix(effts[:, Not(1)])

reload()

r_efffts = HDMjl.rlassoEffects(x, y, index = [1, 2, 3, 4, 5, 16]);
HDMjl.r_summary(r_efffts)


# ----
cps = r_data("cps")
# cps
y = cps[:, 1]
x = Matrix(cps[:, Not(1)])
inx = Array(vcat(1, 13:21))
# [12, 3]

cps_female = HDMjl.rlassoEffects(x, y, index = 1)
size(x)
cps_female.result

HDMjl.r_summary(cps_female)

x[:, inx]

### -------------------------------

dta4 = r_data("cnt")

y = dta4[:, 1]
x = dta4[:, Not(1)]

eff3 = HDMjl.rlassoEffects(Matrix(x), y)
HDMjl.r_summary(eff3)


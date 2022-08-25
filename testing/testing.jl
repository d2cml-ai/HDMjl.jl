using CSV, Test

include("../scr/hdmjl.jl")

## ajr data

AJR = CSV.read("data/ajr.csv", DataFrame)

categorical!(AJR, [:Africa, :Asia, :Namer, :Samer])
y = AJR.GDP
x_form = @formula(GDP ~ Latitude + Latitude2 + Africa + Asia + Namer + Samer)
x = modelmatrix(x_form, AJR)
d = AJR.Exprop
z = AJR.logMort;
y = ajr.GDP

### rlasso normal
rls_nn = rlasso(x, y)["coefficients"]

@test isapprox(rls_nn[1], 8.35, atol =.1) 
@test isapprox(rls_nn[3], 3.79, atol =.1) 
@test isapprox(rls_nn[4], -1.2, atol =.1) 


## rlasso params 

rls_p = rlasso(x, y, intercept = false)

@test isapprox.(rls_p["lambda"], [34.46013, 16.92229, 37.16298, 33.50906, 21.13842, 19.51323], atol = .1)
@test isapprox(rls_p["lambda0"],  50.64526, atol = .1)
@test isapprox.(rls_p["coefficients"], [8.195694, 0.000000, 5.983412, 6.900308, 6.698197, 7.181136])

##- rlasso logit

include("../scr/rlassologit.jl")

x = CSV.read("data/x_logit.csv", DataFrame)
y = CSV.read("data/y_logit.csv", DataFrame)
using CSV, DataFrames, GLM, GLMNet
using Distributions, Random, Statistics

rls_logit = rlassologit(Matrix(x), y.x)
rls_logit["coefficients"]

@test isapprox(rls_logit["coefficients"], [0.6264361, 1.6309953, 2.2731915, 1.6581630, 2.3191826, 2.0848546, 2.6163817,
2.3308691, 1.9528631, 2.5675231, 2.1257439, 0.0000000, 0.0000000, 0.0000000,
0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000], atol = .1)

@test isapprox(rls_logit["lambda0"], 28.85712, atol = .1)

@test sum(Array(rls_logit["index"] .- [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) == 0

#### -- x

rls_logit_i = rlassologit(Matrix(x), y.x, intercept = false)
# rls_logit_p = rlassologit(Matrix(x), y.x, post = false)
@test isapprox(rls_logit_i["coefficients"]', [1.524081 2.256433 1.655706 2.295221 1.886550 2.505409 2.202559 1.941222 2.386040 2.006878 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000], atol = .1)
@test sum(rls_logit_i["index"]' .- [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]) == 0
@test isapprox(rls_logit_p["lambda0"], 28.85, atol = 100)


### rlassologitEffect(s)
logit_effect = rlassologitEffect(Matrix(x[:, 2:20]), y.x, x[:, 1])

@test isapprox(logit_effect["alpha"], 1.631, atol = .1)
@test isapprox(logit_effect["residuals"]["v"][1:5]', [-0.7070339  0.2078543  1.6345662 -1.0842999 -0.0587829], atol = .1)

using StatsBase
# v = sample(1:20, 3) [8, 2, 20]

logit_effects = rlassologitEffects(Matrix(x), y.x, index = v)
@test isapprox(logit_effects["coefficients"]', [1.9528631  2.2731915 -0.2798281], atol = .1)
@test isapprox(logit_effects["se"]', [0.2849660 0.3119032 0.3243001], atol = .08)


### rlassoTreatment

include("../scr/rlassoEffect.jl")

# y = AJR.GDP
# x_form = @formula(GDP ~ Latitude + Latitude2 + Africa + Asia + Namer + Samer)
# x = modelmatrix(x_form, AJR)
# d = AJR.Exprop
# z = AJR.logMort;
# y = ajr.GDP

# y = CSV.read("data/y_rlassoE.csv", DataFrame).V1
# x = CSV.read("data/x_rlassoE.csv", DataFrame) |> Matrix
# df = hcat(y, x)
rlassoE = rlassoEffect(df[:, Not(1)], df[:, 1], df[:, 3])

### FIXME: incorrect residuals
@test isapprox(rlassoE["coefficients"], 3.155, atol = 0.05)


## Rcode example not found
### rlassoATE work  without example R

include("../scr/rlassoTreatment.jl")

d = sample([0, 1], 100)
z = sample([0, 1], 100)
rlassoLATE(x, d, y, z)
rlassoATE(x, d, y)

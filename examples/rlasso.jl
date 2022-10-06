
using CSV, DataFrames
function r_data(n = 1)
    n_m = "examples/r_" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

reload() = include("../src/HDMjl.jl")
### 59 Pass
dta = r_data(1)
n, p = size(dta)
s = 3
beta = vcat(fill(3, s), zeros(p - s));

p = p-1
X = dta[:, Not(1)]
Y = dta[:, 1];
reload()
lasso_reg = HDMjl.rlasso(X, Y, post = false)
HDMjl.r_summary(lasso_reg)

#------
yhat_lasso = HDMjl.r_predict(lasso_reg)
Xnew = r_data("1.1xnew")
yhat_lasso_new = HDMjl.r_predict(lasso_reg, xnew = Matrix(Xnew))
post_lasso_reg = rlasso(X, Y, post = true)
y_hat_postlasso = r_predict(post_lasso_reg, xnew = Matrix(Xnew))
## TODO: Implementar la funcion print para rlasso
r_summary(post_lasso_reg)

ynew = r_data("1.1ynew")

y_hat_postlasso
using Statistics
# ynew
mean(abs.(ynew[:, 1] - yhat_lasso_new)); mean(abs.(ynew[:, 1] - y_hat_postlasso))

#### ----------------

dat2 = r_data(2)

y = dat2[:, 1]
d = dat2[:, 2]
x0 = dat2[:, Not(1, 2)]

rlasso(x0, y)
rlasso(x0, d)



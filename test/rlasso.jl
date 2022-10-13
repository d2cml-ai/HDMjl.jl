### 3.2
# pwd()

reload() = include("../src/HDMjl.jl")

# reload()

using CSV, DataFrames, PrettyTables

function r_data(n = 1)
    n_m = "data/" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

function intercept(X)
    n, p = size(X)
    X1 = hcat(ones(n), X, makeunique = true)
    return Matrix(X1)
end

r_32 = r_data("3.2")

y = r_32[:, 1]
x = r_32[:, Not(1)]
# intercept(x)

reload()
lasso_reg = hdm.rlasso(x, y, post = false)
hdm.r_summary(lasso_reg)

yhat_lasso = hdm.r_predict(lasso_reg)
d_new = r_data("3.2_new")
Xnew = d_new[:, Not(1)]
ynew = d_new[:, 1]
lasso_reg["coefficients"]

yhat_lasso_new = hdm.r_predict(lasso_reg, xnew = Matrix(Xnew))
post_lasso_reg = hdm.rlasso(x, y, post = true)
y_hat_postlasso = hdm.r_predict(post_lasso_reg, xnew = Matrix(Xnew))
## TODO: Implementar la funcion print para rlasso
hdm.r_summary(post_lasso_reg)

using Statistics

y_hat_postlasso
# ynew
mean(abs.(ynew[:, 1] - yhat_lasso_new)), mean(abs.(ynew[:, 1] - y_hat_postlasso))

################# 4
### 4.1

r_41 = r_data(4.1)

x_41 = r_41[:, Not(1)]
y_41 = r_41[:, 1]

using GLM

full_fit = GLM.lm(intercept(x_41), y_41)

est = round(coeftable(full_fit).cols[1][2], digits = 5)
s_td = round(coeftable(full_fit).cols[2][2], digits = 5)
print("Estimate: $est ($s_td)")

d_41 = x_41[:, 1]
X1 = x_41[:, Not(1)] 

lm_y = lm(intercept(X1), y_41)
lm_d = lm(intercept(X1), d_41)
# lm_y
n = size(r_41, 1)
rY = GLM.residuals(lm_y)
rd = GLM.residuals(lm_d)
partial_fit_ls = lm(hcat(ones(n), rd), rY)
est = round(coeftable(partial_fit_ls).cols[1][2], digits = 5)
s_td = round(coeftable(partial_fit_ls).cols[2][2], digits = 5)
print("Estimate: $est ($s_td)")


rY = hdm.rlasso(X1, y_41)["residuals"]
rd = hdm.rlasso(X1, d_41)["residuals"]
# intercept(rd)
# rY

partial_fit_ls = GLM.lm(hcat(ones(n), rd), rY[:, 1])
est = round(coeftable(partial_fit_ls).cols[1][2], digits = 6)
s_td = round(coeftable(partial_fit_ls).cols[2][2], digits = 6)
print("Estimate: $est ($s_td)")

print("\n\n\n")

### rlassoEffect

Eff = hdm.rlassoEffect(x_41[:, Not(1)], y_41, x_41[:, 1], method = "partialling out");
hdm.r_summary(Eff);
reload()
x[:, [1, 2]]


reload()
Eff = hdm.rlassoEffect(x_41[:, Not(1)], y_41, x_41[:, 1], method = "double selection");
hdm.r_summary(Eff);



########## 4.2

r_42 = r_data(4.2)
x_42 = r_42[:, Not(1)]
y_42 = r_42[:, 1]

lassoeffect = hdm.rlassoEffects(x_42, y_42, index = [1, 2, 3, 50]);

hdm.r_print(lassoeffect)
hdm.r_summary(lassoeffect)
hdm.r_confint(lassoeffect)



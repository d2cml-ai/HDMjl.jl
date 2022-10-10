
set.seed(99)
n = 5000
p = 20
X = matrix(rnorm(n * p), ncol = p)
colnames(X) = c("d", paste("x", 1:19, sep = ""))
xnames = colnames(X)[-1]
beta = rep(1, 20)
y = X %*% beta + rnorm(n)

r_data = function (name = "nn", ...){
    nn = paste0("examples/r_", name, ".csv")
    # cbind(...)
    dta = cbind(...)
    write.csv(dta, nn, row.names = F)
    # print(head(dta))
    print(nn)
}
r_data("1_eff", y, X)
library(hdm)

x1 = X[, -1]

Eff = rlassoEffect(X[, -1], y, X[, 1], method = "partialling out")
summary(Eff)
Eff = rlassoEffect(X[, 2:4], y, X[, 1], method = "double selection")
summary(Eff)


GrowthData = hdm::GrowthData
y = GrowthData[, 1, drop = F]
d = GrowthData[, 3, drop = F]
X = as.matrix(GrowthData)[, -c(1, 2, 3)]

varnames = colnames(GrowthData)

xnames = varnames[-c(1, 2, 3)]
# names of X variables
dandxnames = varnames[-c(1, 2)]
# names of D and X variables
# create formulas by pasting names (this saves typing times)
# fmla = as.formula(paste("Outcome ~ ", paste(dandxnames, collapse = "+")))
# ls.effect = lm(fmla, data = GrowthData)

r_data("growth_data", y, d, X)

dX = as.matrix(cbind(d, X))
dim(X)
x1 = X[, 1:60]
lasso.effect = rlassoEffect(x = X, y = y, d = d, method = "partialling out")
summary(lasso.effect)

double.selection = rlassoEffect(x = X, y = y, d = d, method = "double selection")
summary(double.selection)

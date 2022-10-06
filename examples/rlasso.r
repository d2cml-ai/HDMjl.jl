library(hdm)
# 3.2 A Joint Significance test for Lasso Regression
r_data = function (name = "nn", ...){
    nn = paste0("examples/r_", name, ".csv")
    # cbind(...)
    dta = cbind(...)
    write.csv(dta, nn, row.names = F)
    # print(head(dta))
    print(nn)
}
set.seed(12345)
n = 100
#sample size
p = 50
# number of variables
s = 3
# nubmer of variables with non-zero coefficients
X = matrix(rnorm(n * p), ncol = p)
beta = c(rep(5, s), rep(0, p - s))
Y = X %*% beta + rnorm(n)
# cbind(Y, X)

r_data(1, Y, X)

r_d = function(n = 1){
    read.csv(paste0("r_", n, ".csv"))
}
head(r_d())




lasso.reg = rlasso(Y ~ X, post = F)
# use lasso, not-Post-lasso
# lasso.reg = rlasso(X, Y, post=FALSE)
sum.lasso <- summary(lasso.reg, all = FALSE)
# can also do print(lasso.reg, all=FALSE)
head(X)
head(Y)

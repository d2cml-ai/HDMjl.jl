library(hdm)


 #4.2 Inference confidence Intervals and Significance Testing
set.seed(1)
n = 100
#sample size
p = 100
# number of variables
s = 3
# nubmer of non-zero variables
X = matrix(rnorm(n * p), ncol = p)
colnames(X) <- paste("X", 1:p, sep = "")
beta = c(rep(3, s), rep(0, p - s))
y = 1 + X %*% beta + rnorm(n)
data = data.frame(cbind(y, X))
colnames(data)[1] <- "y"

r_data("01Effects", data)

fm = paste("y ~", paste(colnames(X), collapse = "+"))
fm = as.formula(fm)
fm
lasso.effect = rlassoEffects(fm, I = ~X1 + X2 + X3 + X4 + X5 + X16, data = data)
print(lasso.effect)
##3 -- 
library(hdm)

cps2012 = head(hdm::cps2012, 2000)

X <- model.matrix(~-1 + female + female:(widowed + divorced + separated +nevermarried +
hsd08 + hsd911 + hsg + cg + ad + mw + so + we + exp1 + exp2 + exp3) + +(widowed+
divorced + separated + nevermarried + hsd08 + hsd911 + hsg + cg + ad + mw + so +
we + exp1 + exp2 + exp3)^2, data = cps2012)
# dim(X)
# [1] 29217
# 136
X <- X[, which(apply(X, 2, var) != 0)]
# exclude all constant variables
# dim(X)
# [1] 29217
# 116
index.gender <- grep("female", colnames(X))
y <- cps2012$lnw

index.gender
options(scipen = 999)
r_data("cps", y, X)

effects.female <- rlassoEffects(x = X, y = y, index = c(1, 2))
summary(effects.female)
dim(X)

head(X[, index.gender])

#----------------------

set.seed(1)
n = 100
p1 = 20
p2 = 20
D = matrix(rnorm(n * p1), n, p1)

W = matrix(rnorm(n * p2), n, p2)
X = cbind(D, W)
# Regressors
Y = D[, 1] * 5 + W[, 1] * 5 + rnorm(n)

r_data("cnt", Y, X)

rlassoEffects(X, Y, index = c(1:p1))

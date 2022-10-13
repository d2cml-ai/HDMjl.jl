library(hdm)

# 3.2 A Joint Significance test for Lasso Regression
set.seed(1235)
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
r_data = function (name = "nn", ...){
    nn = paste0("data/", name, ".csv")
    # cbind(...)
    dta = cbind(...)
    write.csv(dta, nn, row.names = F)
    # print(head(dta))
}
# r_data()
r_data(name = "3.2", Y, X)

# head(X)


lasso.reg = rlasso(Y ~ X, post = F)
# use lasso, not-Post-lasso
# lasso.reg = rlasso(X, Y, post=FALSE)
sum.lasso <- summary(lasso.reg, all = FALSE)


yhat.lasso = predict(lasso.reg)
#in-sample prediction
Xnew = matrix(rnorm(n * p), ncol = p)
# new X
Ynew = Xnew %*% beta + rnorm(n)
#new Y
r_data("3.2_new", Ynew, Xnew)

yhat.lasso.new = predict(lasso.reg, newdata = Xnew)

post.lasso.reg = rlasso(Y ~ X, post = TRUE)
#now use post-lasso
print(post.lasso.reg, all = FALSE)


yhat.postlasso = predict(post.lasso.reg)
#in-sample prediction
yhat.postlasso.new = predict(post.lasso.reg, newdata = Xnew)
#out-of-sample prediction
MAE <- apply(cbind(abs(Ynew - yhat.lasso.new), abs(Ynew - yhat.postlasso.new)), 2,
mean)
names(MAE) <- c("lasso MAE", "Post-lasso MAE")
print(MAE, digits = 5)


################# 4
### 4.1
set.seed(1)
n = 5000
p = 20
X = matrix(rnorm(n * p), ncol = p)
colnames(X) = c("d", paste("x", 1:19, sep = ""))
xnames = colnames(X)[-1]
beta = rep(1, 20)
y = X %*% beta + rnorm(n)
dat = data.frame(y = y, X)

r_data("4.1", dat)

fmla = as.formula(paste("y ~ ", paste(colnames(X), collapse = "+")))
full.fit = lm(fmla, data = dat)
summary(full.fit)$coef["d", 1:2]


fmla.y = as.formula(paste("y ~ ", paste(xnames, collapse = "+")))
fmla.d = as.formula(paste("d ~ ", paste(xnames, collapse = "+")))
# partial fit via ols
rY = lm(fmla.y, data = dat)$res
rD = lm(fmla.d, data = dat)$res
partial.fit.ls = lm(rY ~ rD)
summary(partial.fit.ls)$coef["rD", 1:2]

rY = rlasso(fmla.y, data = dat)$res
rD = rlasso(fmla.d, data = dat)$res
partial.fit.postlasso = lm(rY ~ rD)
summary(partial.fit.postlasso)$coef["rD", 1:2]

## Rlassoeffect
Eff = rlassoEffect(X[, -1], y, X[, 1], method = "partialling out")
summary(Eff)$coef[, 1:2]

Eff = rlassoEffect(X[, -1], y, X[, 1], method = "double selection")
summary(Eff)$coef[, 1:2]


####### 4.2

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
fm = paste("y ~", paste(colnames(X), collapse = "+"))
fm = as.formula(fm)

r_data("4.2", data)


lasso.effect = rlassoEffects(fm, I = ~X1 + X2 + X3 + X50, data = data)
print(lasso.effect)

summary(lasso.effect)

confint(lasso.effect)

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

head(Y)

lasso.reg = rlasso(Y ~ X, post = F)

# use lasso, not-Post-lasso
# lasso.reg = rlasso(X, Y, post=FALSE)
sum.lasso <- summary(lasso.reg, all = FALSE)


yhat.lasso = predict(lasso.reg)
#in-sample prediction
Xnew = matrix(rnorm(n * p), ncol = p)
r_data("1.1xnew", Xnew)
# new X
Ynew = Xnew %*% beta + rnorm(n)
r_data("1.1Ynew", Ynew)
#new Y
yhat.lasso.new = predict(lasso.reg, newdata = Xnew)
#out-of-sample prediction
post.lasso.reg = rlasso(Y ~ X, post = TRUE)
#now use post-lasso
yhat.postlasso = predict(post.lasso.reg)
#in-sample prediction
yhat.postlasso.new = predict(post.lasso.reg, newdata = Xnew)
#out-of-sample prediction
mean(abs(Ynew - yhat.lasso.new))
mean(abs(Ynew - yhat.postlasso.new))

##################


set.seed(1)
n = 5000
p = 100
p0 = 100 -1
X = matrix(rnorm(n * p), ncol = p)
colnames(X) = c("d", paste("x", 1:p0, sep = ""))
xnames = colnames(X)[-1]
beta = rep(1, p)
y = X %*% beta + rnorm(n)
dat = data.frame(y = y, X)
r_data("2", dat)
# head(dat)
xnames = colnames(X)[-1]
fmla.y = as.formula(paste("y ~ ", paste(xnames, collapse = "+")))
fmla.d = as.formula(paste("d ~ ", paste(xnames, collapse = "+")))

rY = rlasso(fmla.y, data = dat)#$res
rD = rlasso(fmla.d, data = dat)#$res

summary(rD)

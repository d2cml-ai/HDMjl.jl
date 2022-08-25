library(hdm)
library(dplyr)
## Rlasso

### AJR

AJR = hdm::AJR
y = AJR$GDP
x_form = GDP ~ Latitude + Latitude2 + Africa + Asia + Namer + Samer
x = model.matrix(x_form, AJR)
d = AJR$Exprop
z = AJR$logMort

# write.csv(AJR, "data/ajr.csv", row.names = F)
rlasso(y ~ x, intercept = T) |> glimpse()
rls_p = rlasso(y ~ x[, -1], intercept = F) 
rls_p$lambda0 |> as.vector()
rls_p$lambda |> as.vector()
rls_p$coefficients |> as.vector()

help(rlassologit)



set.seed(2)
# n <- 250
# p <- 20
# px <- 10
# X <- matrix(rnorm(n*p), ncol=p)

# X


# head(X)
# beta <- c(rep(2,px), rep(0,p-px))
# intercept <- 1
# P <- exp(intercept + X %*% beta)/(1+exp(intercept + X %*% beta))
# y <- rbinom(nrow(X), size=1, prob=P)

write.csv(X, "data/x_logit.csv", row.names = F)
write.csv(y, "data/y_logit.csv", row.names = F)

## fit rlassologit object
rlassologit.reg <- rlassologit(y~X)
## methods
glimpse(rlassologit.reg)

rlassologit.reg$coefficients |> as.vector()
rlassologit.reg$lambda0
rlassologit.reg$residuals
rlassologit.reg$index |> as.numeric() |> length()

### intercept false
rlassologit.reg_p <- rlassologit(y~X, intercept = F)
## methods
glimpse(rlassologit.reg_p)
X[, 1]
rlassologit.reg_p$coefficients |> as.vector()
rlassologit.reg_p$lambda0
rlassologit.reg_p$residuals
rlassologit.reg_p$index |> as.numeric() #|> length()

# help(rlassologitEffect)

## Not run
xd <- X[,2:20]
d <- X[,1]
logit.effect <- rlassologitEffect(x=xd, d=d, y=y)

glimpse(logit.effect)
logit.effect$residuals$v[1:5, ]
logit.effects <- rlassologitEffects(X,y, index=c(8, 2, 20))
glimpse(logit.effects)

logit.effects$coefficients |> as.vector()
logit.effects$se |> as.vector()

# help(rlassoEffect)

library(hdm); library(ggplot2)
set.seed(1)
n = 100 #sample size
p = 12 # number of variables
s = 3 # nubmer of non-zero variables
X = matrix(rnorm(n*p), ncol=p)

y = 1 + X%*%beta + rnorm(n)
write.csv(X, "data/x_rlassoE.csv", row.names = F)
write.csv(y, "data/y_rlassoE.csv", row.names = F)

# colnames(X) <- paste("X", 1:p, sep="")
beta = c(rep(3,s), rep(0,p-s))
# data = data.frame(cbind(y,X))
colnames(data)[1] <- "y"
# fm = paste("y ~", paste(colnames(X), collapse="+"))
# fm = as.formula(fm)                 
rlassoE = rlassoEffect(X, y, X[, 2])
rlassoE |> glimpse()
lasso.effect = rlassoEffects(X, y, index=c(1,2,3,10))
lasso.effect = rlassoEffects(fm, I = ~ X1 + X2 + X3 + X50, data=data)
# print(lasso.effect)
# summary(lasso.effect)
# confint(lasso.effect)
# plot(lasso.effect)

help(rlassoATE)
d = read.csv("data/y_logit.csv") |> head(100)

z

rlassoATE(x = X, d = d, y = y)
## Default S3 method:
rlassoATE(X, d, y, bootstrap = "none", nRep = 500)
rlassoATE()
rlassoLATE

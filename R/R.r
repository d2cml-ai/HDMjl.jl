library(here)
library(glmnet)

here()
x = read.csv("jl/Data/x_rnd.csv",  header = F)
y = read.csv("jl/Data/y_rnd.csv", header = F)

glmnet(x, y, family = "binomial", alpha = 1, standardize = T, intercept = T)

glmnet()
?glmnet

# binomial
# Gaussian
x = matrix(rnorm(100 * 20), 100, 20)
y = rnorm(100)

write.csv(x, "jl/Data/x_rnd.csv")

# fit1 = glmnet(x, y)
# print(fit1)
# coef(fit1, s = 0.01)  # extract coefficients at a single value of lambda
# predict(fit1, newx = x[1:10, ], s = c(0.01, 0.005))  # make predictions

g2 = sample(c(0,1), 100, replace = TRUE)
write.csv(g2, 'jl/Data/y_rnd.csv')

fit2 = glmnet(x, g2, family = "binomial", lambda = 12)
fit2
# fit2n = glmnet(x, g2, family = binomial(link=cloglog))
# fit2r = glmnet(x,g2, family = "binomial", relax=TRUE)
# fit2rp = glmnet(x,g2, family = "binomial", relax=TRUE, path=TRUE)
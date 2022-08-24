using CSV, DataFrames, GLM, GLMNet
using Distributions, Random, Statistics

# pwd()
# ajr = CSV.read("jl/Data/ajr.csv", DataFrame)

# y = ajr.GDP
# x_form = @formula(GDP ~ Latitude + Latitude2 + Africa + Asia + Namer + Samer)
# x = modelmatrix(x_form, ajr)
# d = ajr.Exprop
# z = ajr.logMort
y = CSV.read("jl/Data/y_rnd.csv", DataFrame)
x = CSV.read("jl/Data/x_rnd.csv", DataFrame)
y = y.x
x = x[:, 2:21]

# params

size(x)




function rlassologit(x, y; 
    penalty_c::Float64 = 0.1,
    post::Bool = true,
    penalty_gamma::Float64 = .1,
    penalty_lambda::Any = nothing, #nothing
    intercept::Bool = true,
    control_threshold::Any = nothing #nothing
    )


    n, p = size(x)


    if !isnothing(penalty_c)
        if post
            penalty_c = 1.1
        else
            penalty_c = .5
        end
    end

    if isnothing(penalty_gamma) 
        penalty_gamma = 0.1 / log(n)
    end


    if !isnothing(penalty_lambda)
        lambda = penalty_lambda / (2 * n)
        lambda0 = lambda * (2 * n)
    else
        lambda0 = penalty_c / 2 * sqrt(n) *quantile(Normal(0.0, 1.0), 1 - penalty_gamma/(2*p))
        lambda = lambda0 / (2 * n)
    end
    # lambda
    s0 = sqrt(var(y))

    x1 = Matrix(x)
    # y1 = Matrix(y[:, :])


    log_lasso = glmnet(x1, y, lambda = [lambda[1]], alpha = 1.0, intercept = true, standardize = true)
    coefTemp = log_lasso.betas

    for i in eachindex(coefTemp)
        if isnan(coefTemp[i])
            coefTemp[i] = 0
        end
    end

    ind = abs.(coefTemp) .> 0

    columns_select = []

    for i in eachindex(ind)
        if ind[i] == 1
            append!(columns_select, i)
        end
    end
    x2 = x1[:, columns_select]
    if size(x2, 2) == 0
        if intercept == true
            a0 = log(mean(y) / (1 - mean(y)))
            res = y .- mean(y)
            coef = append!([a0], repeat([0], p))
        end
        if intercept == false
            a0 = 0
            res = y .- 0.5
            coef = repeat([0], p)
        end

        index = repeat([false], p)

        est = Dict(
            "coefficients" => coef, "beta" => coefTemp, "intercept" => a0,
            "index" => index, "s0" => s0, "lambda0" => lambda0, "residuals" => res,
            "sigma" => sqrt(var(res))
        )

        return est
    #  return Dict("lambda0" => lmbda0, "lambda" => lmbda, "Ups0" => Ups0) 
    end

    coefTemp1 = Matrix(coefTemp)
    df = DataFrame(hcat(y, x2))
    rename!(df, :x1 => "y")

    if post
        if intercept
            reg = glm(Term(:y) ~  sum(Term.(Symbol.(names(df[:, Not(:y)])))), df, Binomial()) 
            coefT = GLM.coef(reg)[2:end]
            for i in eachindex(coefT)
                if isnan(coefT[i])
                    coefT[i] = 0
                end
            end
            e1 = y - GLM.predict(reg)
            coefTemp1[columns_select] = coefT
            # for i in eachindex(ind)
        end
        if !intercept
            reg = glm(Term(:y) ~  ConstantTerm(0) + sum(Term.(Symbol.(names(df[:, Not(:y)])))), df, Binomial()) 
            coefT = GLM.coef(reg)
            for i in eachindex(coefT)
                if isnan(coefT[i])
                    coefT[i] = 0
                end
            end
            e1 = y - GLM.predict(reg)
            coefTemp1[columns_select] = coefT
        end
    elseif !post
        e1 = y - GLMNet.predict(log_lasso, x1, outtype = :response)
    end


    ctr_inx = []
    if !isnothing(control_threshold)
        ctr_eval = abs.(coefTemp1) .< control_threshold
        for i in eachindex(ctr_eval)
            if ctr_eval[i] == 1
                append!(ctr_inx, 0)
            else
                append!(ctr_inx, coefTemp[i])
            end
        end
    end


    if intercept
        if post == true
            a0 = GLM.coef(reg)[1]
        end
        if post == false
            a0 = log_lasso.a0
        end
        coefs = vcat(a0, coefTemp1)
    else
        a0 = 0
        coefs = coefTemp1
    end

    est = Dict(
        "coefficients" => coefs, "beta" => coefTemp1, "intercept" => a0, "index" => ind,
        "lambda0" => lambda0, "residuals" => e1, "sigma" => sqrt(var(e1)),
        "options" => Dict("post" => post, "intercept" => intercept, "control" => control_threshold)
    )
    return est

end
    
r = rlassologit(x, y, post = false)
# r["beta"]

### rlassologitEffect

x
y
d
I3 = []
post = true

x = Matrix(x[:, :])
d = Matrix(d[:, :])
y = Matrix(y[:, :])

n, p = size(x)

la1 = 1.1 / 2 & sqrt(n) * quantile(Normal(0, 1), 1 - 0.05 / (max(n, (p + 1) * log(n))))

dx = hcat(d, x)

l1 = rlassologit(dx, y, post = post, intercept = true, penalty_lambda = la1)
x1 = l1["residuals"]

t = l1["coefficients"] * dx

sigma2 = exp.(t) ./ (1 .+ exp.(t)).^2

w = copy(sigma2)
f = sqrt.(sigma2)

I1 = l1["index"]

lambda = 2.2 * sqrt(n) * quantile(Normal(0, 1), 1 - 0.05 / max(n, p * log(n)))
la2 = repeat([lamda], p)
xf = x .* f
df = d .* f

l2 = rlasso(xf, df, post = post, intercept = true, homocedastic = "none", lambda_start = la2, penalty_c = 1.1, penalty_gamma = 0.1)

I2 = l2["index"]
z = l2["residuals"] ./ sqrt.(sigma2)

if unique(I3) == 2
    I = I1 + I2 + I3
    # I = as_logical(I)
else
    I = I1 + I2
    # I = as_logical(I)
end

ind = []
for i in eachindex(I)
    if i > 1
        append!(ind, i)
    end
end


xselect = x[:, I]
p3 = size(xselect, 2)

data3 = hcat(y, d, xselect)
rename!(data3, x1 => "y")


l3 = glm(Term(:y) ~  sum(Term.(Symbol.(names(data3[:, Not(:y)])))), data3, Binomial()) 
alpha = GLM.coef(l3)[2]

t3 = GLM.predict(l3)
g3 = exp.(t3) / (1 + exp.(t3))
w3 = g3 * (1 - g3)
s21 = 1 / mean(@. w3 * d * z)^2 * mean((y - g3).^2 * z.^2)
# TODO: index GLM
# xtilde = x[:, index(l3)]
# p2 = sum(index(l3))

b = hcat(d, xtilde)
p3 = size(b, 2)
A = zeros(p4, p4)

for i in i:n
    w3[i]  #*outer(b[i, :], b[i, :])
end

# TODO: 210 en adelante
s22 = 
function rlassologit(x, y; model::Bool = true, c::Float64 = 1.1, post::Bool = true, 
    n::Int64 = size(x, 1), gamma::Float64 = 0.1 / log(n), 
    lambda::Any = nothing, intercept::Bool = true, 
    threshold::Any = nothing)
    

    

    n = size(x, 1)
    p = size(x, 2)
    if !isnothing(c)
        if post
            c = 1.1
        else
            c = .5
        end
    end

    if isnothing(gamma) 
        gamma = 0.1 / log(n)
    end


    if !isnothing(lambda)
        lambda = lambda / (2 * n)
        lambda0 = lambda * (2 * n)
    else
        lambda0 = c / 2 * sqrt(n) *quantile(Normal(0.0, 1.0), 1 - gamma/(2*p))
        lambda = lambda0 / (2 * n)
    end
    
    if isnothing(threshold)
        threshold = 0
    end

    s0 = sqrt(var(y))
    log_lasso = glmnet(x, y, lambda = [lambda[1]], alpha = 1.0, intercept = intercept, standardize = true) # output issue comes from glmnet behavior in R
    coefTemp = vec(convert(Matrix, log_lasso.betas))
    coefTemp[isnan.(coefTemp)] .= 0
    ind1 = abs.(coefTemp) .> 0
    x1 = x[:, ind1]
    
    if isempty(x1)
        if intercept
            a0 = log(mean(y) / (1 - mean(y)))
            res = y .- mean(y)
            coefs = vcat(a0, zeros(p))
        else
            a0 = 0
            res = y .- 0.5
            coefs = zeros(p)
        end
        est = Dict("coefficients" => coefs, "beta" => coefTemp, "intercept" => a0, 
        "index" => zeros(Bool, p), "s0" => s0, "lambda0" => lambda0, "residuals" => res, 
        "sigma" => sqrt(var(res)), "options" => Dict("post" => post, "intercept" => intercept, "threshold" => threshold))

        return est 
    end

    if post
        if intercept
            reg = glm(hcat(ones(n), x1), y, Binomial(), LogitLink())
            coefT = GLM.coef(reg)[Not(1)]
            coefT[isnan.(coefT)] .= 0
            e1 = y - GLM.predict(reg)
            coefTemp[ind1] .= coefT
        elseif !intercept
            reg = glm(x1, y, Binomial(), LogitLink())
            coefT = GLM.coef(reg)
            coefT[isnan.(coefT)] .= 0
            e1 = y - GLM.predict(reg)
            coefTemp[ind1] = coefT
        end
    elseif !post
        e1 = y - GLMNet.predict(log_lasso, x, outtype = :response)
    end
    
    coefTemp[abs.(coefTemp) .< threshold] .= 0

    if intercept
        if post == true
            a0 = GLM.coef(reg)[1]
        end
        if post == false
            a0 = log_lasso.a0
        end
        coefs = vcat(a0, coefTemp)
    elseif !intercept
        a0 = 0
        coefs = coefTemp
    end
    
    
    ### === Output print
    head1 = "
    Post-Lasso estimation: $post
    Intercept: $intercept
    Control: $threshold
    \n 
    Total number of variables: $p\n
    Number of selected variables: \n
    "
    print(head1)
    if intercept names_columns = [] else names_columns = [] end
    select_columns = []

    for i in eachindex(coefs)
        vl = "V $i"
        push!(names_columns, vl)
        if coefs[i] != 0
            push!(select_columns, i)
        end
    end

    table_lgt = hcat(names_columns, coefs)
    @ptconf tf = tf_simple alignment = :r
    header = ["Variable", "Estimate"]
    @pt :header = header table_lgt

    print("rlassologit")
    # print("head")
    ###
    
    est = Dict("coefficients" => coefs, "beta" => coefTemp, "intercept" => a0, "index" => ind1,
    "lambda0" => lambda0, "residuals" => e1, "sigma" => sqrt(var(e1)),
    "options" => Dict("post" => post, "intercept" => intercept, "control" => threshold));


    return est;
end


function rlassologitEffect(X, Y, D; I3::Any = nothing, post = true)

    x = Matrix(X[:, :])
    d = D[:]
    y = Y[:]
    # d = bi[:, 3]
    # y = bi[:, 2]

    n, p = size(x)

    la1 = 1.1 / 2 * sqrt(n) * quantile(Normal(0, 1), 1 - 0.05 / (max(n, (p + 1) * log(n))))

    dx = hcat(d, x)

    l1 = rlassologit(dx, y, post = post, intercept = true, lambda = la1)
    x1 = l1["residuals"]

    t = hcat(ones(n), dx) * l1["coefficients"]

    sigma2 = exp.(t) ./ (1 .+ exp.(t)).^2

    w = copy(sigma2)
    f = sqrt.(sigma2)

    I1 = l1["ind1ex"][Not(1)]

    lambda = 2.2 * sqrt(n) * quantile(Normal(0, 1), 1 - 0.05 / max(n, p * log(n)))
    la2 = repeat([lambda], p)
    xf = x .* f
    df = d .* f
    l2 = rlasso(xf, df, post = post, intercept = true, homoskedastic = false, lambda_start = la2, c = 1.1, gamma = 0.1)
    # return l2
    # include("hdmjl.jl")
    I2 = l2["ind1ex"]
    z = l2["residuals"] ./ sqrt.(sigma2)

    if isnothing(I3)
        I = I1 + I2 
        I = as_logical(I)
    else
        I = I1 + I2 + I3
        I = as_logical(I)
    end


    ind1 = []
    for i in eachind1ex(I)
        if I[i] > 0 
            append!(ind1, i)
        end
    end
    ind1

    xselect = x[:, ind1]
    p3 = size(xselect, 2)

    data3 = DataFrame(hcat(y, d, xselect))
    rename!(data3, "x1" => "y")


    l3 = glm(Term(:y) ~  sum(Term.(Symbol.(names(data3[:, Not(:y)])))), data3, Binomial()) 
    alpha = GLM.coef(l3)[2]
    data3
    g3 = GLM.predict(l3)
    w3 = @. g3 * (1 - g3)
    s21 = 1 / mean(@. w3 * d * z)^2 * mean((y .- g3).^2 .* z.^2)
    # TODO: ind1ex GLM
    # xtilde = x[:, ind1ex(l3)]
    # p2 = sum(ind1ex(l3))
    S2 = max(s21)
    se = sqrt(s21 / n)
    tval = alpha / se
    if isnothing(I)
        no_select = 1
    else
        no_select = 0
    end
    # GLM.residuals(l3)

    res = Dict("epsilon" => y - g3, "v" => z)
    results = Dict(
        "alpha" => alpha, "se" => se, "t" => tval,# "pval" => pval,
        "no_select" => no_select, "coefficients" => alpha, "coefficient" => alpha,
        "residuals" => res, "sample_size" => n, "post" => post
    )
    return results
end

function rlassologitEffects(x, y; ind1ex = 1:3, I3 = nothing, post = true)
    x = Matrix(x)
    y = Matrix(y[:, :])
    n, p = size(x)
    
    if Set(ind1ex) == 2
        k = p1 = sum(ind1ex)
    else
        k = p1 = length(ind1ex)
    end

    coefficients = zeros(k)
    se = zeros(k)
    t = zeros(k)
    reside = Dict("epsilon" => Dict(), "v" => Dict())
    lasso_regs = Dict()

    # print(k)
    for i in 1:k
        d = x[:, ind1ex[i]]
        xt = x[:, Not(ind1ex[i])]
        if isnothing(I3)
            I3m = I3
        else
            I3m = I3[Not(ind1ex[i])]
        end
        
        lasso_regs[i] = try
            rlassologitEffect(xt, y, d, I3 = I3m, post = post)
        catch
            "try-error"
        end
        if lasso_regs[i] == "try-error"
            continue
        else
            coefficients[i] = lasso_regs[i]["alpha"]
            se[i] = lasso_regs[i]["se"]
            t[i] = lasso_regs[i]["t"]
            reside["epsilon"][i] = lasso_regs[i]["residuals"]["epsilon"]
            reside["v"][i] = lasso_regs[i]["residuals"]["v"]
        end
    end
    # residual = 
    res = Dict(
        "coefficients" => coefficients, "se" => se, "t" => t,
        "lasso_regs" => lasso_regs, "ind1ex" => ind1ex, "sample_size" => n,
        "residuals" => reside
    )
    return res
end
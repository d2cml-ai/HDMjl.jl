function rlasso(x, y; post = true, intercept = true, model = true, 
        homoskedastic = false, X_dependent_lambda = false, lambda_start = nothing, 
        c = 1.1, maxIter = 15, tol::Float64 = 1e-5, n = size(y, 1), gamma = 0.1 / log(n), threshold = nothing)
    if isnothing(threshold)
        threshold = 0
    end
    n = size(y, 1)
    p = size(x, 2)
    
    if intercept
        meanx = mean(x, dims = 1)
        x = x .- meanx
        mu = mean(y)
        y = y .- mu
    else
        meanx = zeros(1, p)
        mu = 0
    end
    
    XX = x' * x
    Xy = x' * y
    
    Psi = mean.(eachcol(x .^2))
    ind = zeros(Bool, p)
    
    startingval = init_values(x, y)["residuals"]
    pen = lambdaCalculation(x = x, y = startingval, homoskedastic = homoskedastic, X_dependent_lambda = X_dependent_lambda, lambda_start = lambda_start, c = c, gamma = gamma)
    lambda = pen["lambda"]
    Ups0 = Ups1 = pen["Ups0"]
    lambda0 = pen["lambda0"]
    
    mm = 1
    s0 = sqrt(var(y))
    
    while mm <= maxIter
        if mm == 1 & post
            coefTemp = LassoShooting_fit(x, y, lambda / 2, XX = XX, Xy = Xy)["coefficients"]
        else
            coefTemp = LassoShooting_fit(x, y, lambda, XX = XX, Xy = Xy)["coefficients"]
        end
        
        global coefTemp[isnan.(coefTemp)] .= 0
        global ind1 = abs.(coefTemp) .> 0
        global x1 = x[:, ind1]
        
        if isempty(x1)
            if intercept
                intercept_value = mean(y .+ mu)
                coef = zeros(p + 1, 1)
            else
                intercept_value = mean(y)
                coef = zeros(p, 1)
            end
            est = Dict("coefficients" => coef, "beta" => zeros(p, 1), "intercept" => intercept_value, "index" => zeros(Bool, p), "lambda" => lambda, "lambda0" => lambda0, "loadings" => Ups0, 
            "residuals" => y .- mean(y), "sigma" => var(y), "iter" => mm, "options" => Dict("post" => post, "intercept" => intercept, "ind_scale" => ind, "mu" => mu, "meanx" => meanx))
            if model
                est["model"] = x
            else
                est["model"] = nothing
            end
            est["tss"] = est["rss"] = sum((y .- mean(y)) .^ 2)
            est["dev"] = y .- mean(y)
            return est
        end
        
        if post
            data_post = hcat(y, x1)
            reg = lm(data_post[:, Not(1)], data_post[:, 1])
            coefT = GLM.coef(reg)
            coefT[isnan.(coefT)] .= 0
            global e1 = y - x1 * coefT
            coefTemp[ind1] = coefT
        elseif !post
            e1 = y - x1 * coefTemp[ind1]
        end 
        s1 = sqrt(var(e1))
        
        # Homoskedastic and X-independent
        if homoskedastic == true & !X_dependent_lambda
            Ups1 = s1 * Psi
            lambda = pen["lambda0"] * Ups1
        
        # Homoskedastic and X-dependent
        elseif homoskedastic == true & X_dependent_lambda
            Ups1 = s1 * Psi
            lambda = pen["lambda0"] * Ups1
            
        # Heteroskedastic and X-independent
        elseif homoskedastic == false & !X_dependent_lambda
            Ups1 = 1 / sqrt(n) .* sqrt.(((e1 .^ 2)' * (x .^ 2))')
            lambda = Ups1 * pen["lambda0"]
            
        # Heteroskedastic and X-dependent
        elseif homoskedastic == false & X_dependent_lambda
            lc = lambdaCalculation(x = x, y = e1, homoskedastic = homoskedastic, X_dependent_lambda = X_dependent_lambda, lambda_start = lambda_start, c = c, gamma = gamma)
            Ups1 = lc["Ups0"]
            lambda = lc["lambda"]

        # Homoskedastic = "none"
        elseif homoskedastic == "none"
            if isnothing(lambda_start)
                throw(ArgumentError("lambda_start required when homoskedastic is set to none" ))
            end
            Ups1 = 1 / sqrt(n) .* sqrt.(((e1 .^ 2)' * (x .^ 2))')
            lambda = Ups1 .* pen["lambda0"]
        end

        mm = mm + 1
        if abs(s0 - s1) < tol
            break
        end
        s0 = s1
    end
    
    if isempty(x1)
        coefTemp = None
        ind1 = zeros(p)
    end
    global coefTemp = coefTemp
    coefTemp[abs.(coefTemp) .< threshold] .= 0
    global ind1 = ind1
    if intercept
        if isnothing(mu)
            mu = 0
        end
        if isnothing(meanx)
            meanx = zeros(size(coefTemp, 1))
        end
        if sum(ind) == 0
            intercept_value = mu - sum(meanx * coefTemp)
        else
            intercept_value = mu - sum(meanx * coefTemp)
        end
    else
        intercept_value = nothing
    end
    
    if intercept
        beta = vcat(intercept_value, coefTemp)
    else
        beta = coefTemp
    end
    
    s1 = sqrt(var(e1))
    est = Dict("coefficients" => beta, "beta" => coefTemp, "intercept" => intercept_value, "index" => ind1, 
        "residuals" => e1, "sigma" => s1, "loadings" => Ups1, "iter" => mm, "lambda0" => lambda0, "lambda" => lambda, 
        "options" => Dict("post" => post, "intercept" => intercept, "ind_scale" => ind, "mu" => mu, "meanx" => meanx), "model" => model)
    if model
        x = x .+ meanx
        est["model"] = x
    else
        est["model"] = nothing
    end
    est["tss"] = sum((y .- mean(y)) .^ 2)
    est["rss"] = sum(est["residuals"] .^ 2)
    est["dev"] = y .- mean(y)
    return est
end

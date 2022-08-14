function rlasso(x, y; 
        post::Bool = true, 
        intercept::Bool = true, 
        model::Bool = true, 
        homoskedastic::Bool = true, 
        X_dependent_lambda::Bool = false, 
        lambda_start::Any = nothing, 
        c::Float64 = 1.1, 
        n::Int64 = size(y, 1),
        gamma::Float64 = 0.1 / log(n), 
        maxIter::Int64 = 15, 
        tol::Float64 = 1e-5, 
        threshold = nothing,
        colnames = nothing)
    
    n = size(y, 1)
    p = size(x, 2)
    if typeof(x) == DataFrame
        colnames = names(x)
        x = Matrix(x)
    elseif !isnothing(colnames)
        colnames = colnames
    else
        colnames = map.(string, "V", 1:p)
    end
    if typeof(y) == DataFrame
        y = Matrix(y)
    end
    
    if intercept
        meanx = mean(x, dims = 1)
        x = x .- meanx
        mu = mean(y)
        y = y .- mu
    else
        meanx = zeros(1, p)
        mu = 0
    end
    
    normx = sqrt.(var(x, corrected = true, dims = 2))
    Psi = mean.(eachcol(x.^2))
    ind = zeros(Bool, p)
    
    XX = x'*x
    Xy = x'*y
    
    startingval = init_values(x, y)["residuals"]
    pen = lambdaCalculation(x = x, y = y, homoskedastic = homoskedastic, X_dependent_lambda = X_dependent_lambda, lambda_start = lambda_start, c = c, gamma = gamma)
    lambda = pen["lambda"]
    Ups0 = pen["Ups0"]
    lambda0 = pen["lambda0"]
    
    mm = 1
    s0 = sqrt(var(y, corrected = true))
    y = vec(y)
    
    while mm <= maxIter
        if mm == 1 & post
            global coefTemp = LassoShooting_fit(x, y, lambda ./ 2, XX = XX, Xy = Xy)["coefficients"]
        else
            global coefTemp = LassoShooting_fit(x, y, lambda, XX = XX, Xy = Xy)["coefficients"]
        end
        
        global coefTemp[isnan.(coefTemp)] .= 0
        global ind1 =  abs.(coefTemp) .> 0
        global x1 = x[:, ind1]
        if isnothing(x1)
            if intercept
                intercept_value = mean(y .+ mu)
                coefs = zeros(p+1, 1)
                coefs = DataFrame([append!(["Intercept"], colnames), coefs], :auto)
            else
                intercept_value = mean(y)
                coefs = zeros(p, 1)
                coefs = DataFrame([colnames, coefs], :auto)
            end
            global est = Dict("coefficients"=> coefs,
                    "beta"=> zeros(p, 1),
                    "intercept"=> intercept_value,
                    "index"=> DataFrame([ colnames, zeros(Bool, p) ], :auto),
                    "lambda"=> lambda,
                    "lambda0"=> lambda0,
                    "loadings"=> Ups0,
                    "residuals"=> y .- mean(y),
                    "sigma"=> var(y, corrected = true, dims = 1),
                    "iter"=> mm,
                    #"call"=> Not a Python option
                    "options"=> Dict("post"=> post, "intercept"=> intercept,
                                "ind.scale"=> ind, "mu"=> mu, "meanx"=> meanx)
                )
            if model
                    est["model"] = x
            else
                est["model"] = nothing
            
            end 
            est["tss"] = sum((y .- mean(y)).^2)
            est["rss"] = sum((y .- mean(y)).^2)
            est["dev"] = y .- mean(y)
        end
        if post
            reg = lm(x1, y)
            coefT = coef(reg)
            coefT[isnan.(coefT)] .= 0
            global e1 = y - x1 * coefT
            coefTemp[ind1] = coefT
        elseif !post
            global e1 = y - x1 * coefTemp[ind1]
        end
        s1 = sqrt(var(e1, corrected = true))
        
        # Homoskedastic and X-independent
        if homoskedastic & !X_dependent_lambda
            Ups1 = s1 * Psi
            lambda = pen["lambda0"] * Ups1
        
        # Homoskedastic and X-dependent
        elseif homoskedastic & X_dependent_lambda
            Ups1 = s1 * Psi
            lambda = pen["lambda0"] * Ups1
            
        # Heteroskedastic and X-independent
        elseif !homoskedastic & !X_dependent_lambda
            Ups1 = 1 / sqrt(n) .* sqrt.(((e1' .^ 2) * (x .^ 2))')
            lambda = pen["lambda0"] .* Ups1
            
        # Heteroskedastic and X-dependent
        elseif !homoskedastic & X_dependent_lambda
            lc = lambdaCalculation(homoskedastic = homoskedastic, X_dependent_lambda = X_dependent_lambda, lambda_start = lambda_start, c = c, gamma = gamma)
            Ups1 = lc["Ups0"]
            lambda = lc["lambda"]
            
        # None
        elseif isnothing(homoskedastic)
            Ups1 = 1 / sqrt(n) .* sqrt.(((e1' .^ 2) * (x .^ 2))')
            lambda = pen["lambda0"] .* Ups1
        end
        
        mm = mm + 1
        if abs.(s0 - s1) < tol
            break
        end
        s0 = s1
    end
    
    global Ups1, ind1, coefTemp = Ups1, ind1, coefTemp
    if isnothing(x)
        coefTemp = nothing
        ind1 = zeros(p, 1)
    end    
    if !isnothing(threshold)
        coefTemp[abs.(coefTemp) .< threshold] = 0
    end
    if intercept        
        if isnothing(mu)
            mu = 0
        end
        if isnothing(meanx)
            meanx = zeros( size(coefTemp)[1], 1)
        end
        if sum(ind) == 0
            intercept_value = mu - sum(meanx .* coefTemp)
        else
            intercept_value = mu - sum(meanx .* coefTemp)
        end
    else
        intercept_value = NaN
    end

    if intercept
        beta = vcat(intercept_value, coefTemp)
        beta = DataFrame([ append!(["Intercept"], colnames), beta ], :auto)
    else
        beta = coefTemp
    end
    
    s1 = sqrt(var(e1, corrected = true))
    est = Dict(
    "coefficients" => beta,
    "beta" => DataFrame([colnames, coefTemp], :auto), 
    "intercept" => intercept_value,
    "index" => ind1,
    "lambda" => DataFrame([colnames, vec(lambda)], :auto),
    "lambda0" => lambda0,
    "loadings" => Ups1,
    "residuals" => e1,
    "sigma" => s1,
    "iter" => mm,
    #"call"=> Not a Python option
    "options" => Dict("post" => post, "intercept" => intercept, 
            "ind.scale" => ind, "mu" => mu, "meanx" => meanx),
    "model" => model
    )
    if model
        x = x .+ meanx
        est["model"] = x
    else
        est["model"] = nothing
    end
    return est
end
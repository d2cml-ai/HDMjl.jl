function init_values(x, y; number::Int64 = 5, intercept::Bool = true)


    n, p = size(x)

    corr = []
    for i in 1:p
        append!(corr, abs.(cor(y, x[:, i])))
    end

    print(corr)

    index = sortperm(corr, rev = true)[1 : min(number, p)]
    coefficients = zeros(p)
    data = hcat(y, ones(n), Matrix(x)[:, index])
    

    if intercept
        data = hcat(y, ones(n), Matrix(x)[:, index])
        reg = GLM.lm(data[:, Not(1)], data[:, 1])
        coefficients[index] = GLM.coef(reg)[2:end]
        y_hat = GLM.predict(reg, data[:, Not(1)])
        e = y .- y_hat
    else
        data = hcat(y, Matrix(x)[:, index])
        reg = GLM.lm(data[:, Not(1)], data[:, 1])
        coefficients[index] = GLM.coef(reg)
        y_hat = GLM.predict(reg, data[:, Not(1)])
        e = y .- y_hat   
    end

    res = Dict(
        "coefficients" => coefficients, "residuals" => e, "index" => index, 
        "Data" => data, "predict" =>  y_hat
    )
    return res
end



function lambdaCalculation(; homoskedastic::Union{Bool, String} = false, X_dependent_lambda::Bool = false, lambda_start = nothing, c::Float64 = 1.1, gamma::Float64 = 0.1, numSim::Int = 5000, y = nothing, x = nothing)
    
    # homoskedastic and X-independent
    if homoskedastic == true & !X_dependent_lambda
        p = size(x, 2)
        n = size(x, 1)
        lambda0 = 2 * c * sqrt(n) * quantile(Normal(0.0, 1.0), 1 - gamma / (2 * p))
        Ups0 = sqrt(var(y))
        lambda = zeros(p) .+ lambda0 * Ups0
    
    # homoskedastic and X-dependent
    elseif homoskedastic == true & X_dependent_lambda
        p = size(x, 2)
        n = size(x, 1)
        R = numSim
        sim = zeros(R, 1)
        
        psi = mean(x .^ 2, dims = 1)
        tXtpsi = x ./ sqrt.(psi)
        
        for i in 1:R
            g = reshape(repeat(randn(n), inner = p),(p, n))'
            sim[i] = n * maximum(2 * abs.(mean(tXtpsi, dims = 1)))
        end
        
        lambda0 = 2 * c * sqrt(n) * quantile(Normal(0.0, 1.0), 1 - gamma / (2 * p))
        Ups0 = 1 / sqrt(n) * sqrt.(((y .^ 2)' * (x .^ 2))')
        lambda = lambda0 * Ups0
        
    # heteroskeddastic and X-independent
    elseif homoskedastic == false & !X_dependent_lambda
        p = size(x, 2)
        n = size(x, 1)
        lambda0 = 2 * c * sqrt(n) * quantile(Normal(0.0, 1.0), 1 - gamma / (2 * p))
        Ups0 = 1 / sqrt(n) * sqrt.(((y .^ 2)' * (x .^ 2))')
        lambda = lambda0 * Ups0
        
    # heteroskedastic and X-dependent
    elseif homoskedastic == false & X_dependent_lambda
        p = size(x, 2)
        n = size(x, 1)
        R = numSim
        eh = y
        ehat = reshape(repeat(eh, inner = p), (p, n))'
        xehat = x .* ehat
        psi = mean(xehat .^ 2, dims = 1)
        tXehattpsi = (xehat./sqrt.(psi))
        
        for i in 1:R
            g = reshape(repeat(randn(n), inner = p),(p, n))'
            sim[l] = n * maximum(2*abs.(mean.(eachcol(tXtpsi.* g))))
        end
        
        lambda0 = c*quantile(vec(sim), 1 - gamma)
        Ups0 = (1 /sqrt(n)) * sqrt.((y.^2)'*(x.^2))
        lambda = lambda0 * Ups0
    end
    
    if !isnothing(lambda_start)
        p = size(x, 2)
        if size(lambda_start) == 1
            lambda_start = zeros(p) .+ lambda_start
        end
        lambda = lambda_start
    end

    if homoskedastic == "none"
        if isnothing(lambda_start)
            throw(ArgumentError("lambda_start required when homoskedastic is set to none" ))
        end
        n = size(x, 1)
        lambda0 = lambda_start
        Ups0 = (1 /sqrt(n)) * sqrt.((y.^2)'*(x.^2))
        lambda = lambda0 .* Ups0
    end
    
    return Dict("lambda" => lambda, "lambda0" => lambda0, "Ups0" => Ups0)
end

function as_logical(array)
    b = []
    for i in array
        if i > 0
            append!(b, true)
        else
            append!(b, false)
        end
    end
    return b
end
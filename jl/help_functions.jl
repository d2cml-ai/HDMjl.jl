using Statistics, GLM, LinearAlgebra


function init_values(x, y; number::Int64 = 5, intercept::Bool = true)

    n, p = size(x)

    corr = []
    for i in 1:p
        append!(corr, abs.(cor(y, x[:, i])))
    end

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
    # val = Dict("x" => x1, "y" => y, "ind" => index)
    return res
    # return val
end


function lambdaCalculation(     ; homoskedastic::Bool=false, X_dependent_lambda::Bool=false,
                                lambda_start=nothing, c::Float64=1.1, gamma::Float64=0.1, 
                                numSim::Int=5000, y=nothing, x=nothing, par::Bool=true, 
                                corecap::Float64=Inf, fix_seed::Bool=true)
    
    n, p = size(x)

    # Get number of simulations to use (if simulations are necessary)
    R = numSim

    # Go through all possible combinations of homoskedasticy/heteroskedasticity
    # and X-dependent or independent error terms. The first two cases are
    # special cases: Handling the case there homoskedastic was set to None, and
    # where lambda_start was provided.
    #
        
    # 1) If homoskedastic was set to None (special case)
    if (isnothing(homoskedastic))

        # Initialize lambda
        lmbda0 = lambda_start

        Ups0 = (1 /sqrt(n)) * sqrt.((x.^2)' * (y.^2))

        # Calculate the final vector of penalty terms
        lmbda = lmbda0 * Ups0

    # 2) If lambda_start was provided (special case)
    elseif (isnothing(lambda_start)) == 0
        
        # Check whether a homogeneous penalty term was provided (a scalar)
        if maximum(size(lambda_start)) == 1
            # If so, repeat that p times as the penalty term
            lmbda = ones(p,1).*lambda_start

        else
            # Otherwise, use the provided vector of penalty terms as is
            lmbda = lambda_start
        end

    # 3) Homoskedastic and X-independent
    elseif homoskedastic == true &&  X_dependent_lambda == false

        # Initilaize lambda
        lmbda0 = 2 * c * sqrt(n) * quantile(Normal(0.0, 1.0),1 - gamma/(2*p))

        # Use ddof=1(corrected = true in Julia) to be consistent with R's var() function (in Julia by defaul the DDF is N-1)
        Ups0 = sqrt(var(y, corrected = true))

        # Calculate the final vector of penalty terms
        lmbda = zeros(p,1) .+ lmbda0 * Ups0

    # 4) Homoskedastic and X-dependent
    elseif homoskedastic == true && X_dependent_lambda == true

        psi = mean.(eachcol(x.^2))
        tXtpsi = (x' ./ sqrt(psi))' 

        R = 5000
        sim = zeros(R,1)

        for l in 1:R
                g = reshape(repeat(randn(n), inner = p),(p, n))'
                sim[l] = n * maximum(2*abs.(mean.(eachcol(tXtpsi.* g))))
        end

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c*quantile.(vec(sim), 1 - gamma)

        Ups0 = sqrt(var(y, corrected = true))

        # Calculate the final vector of penalty terms
        lmbda = zeros(p,1) .+ lmbda0 .* Ups0

    # 5) Heteroskedastic and X-independent
    elseif homoskedastic == false &&  X_dependent_lambda == false

        # The original includes the comment, "1=num endogenous variables"
        lmbda0 = 2 * c * sqrt(n) * quantile(Normal(0.0, 1.0),1 - gamma/(2*p*1))

        Ups0 = (1 /sqrt(n)) * sqrt.((y.^2)'*(x.^2))'
        
        lmbda = lmbda0 * Ups0
        
    # 6) Heteroskedastic and X-dependent
    elseif homoskedastic == false &&  X_dependent_lambda == true

        eh = y
        ehat = reshape(repeat(eh, inner = p),(p, n))'

        xehat = x.*ehat
        psi = mean.(eachcol(xehat.^2))'
        tXehattpsi = (xehat./sqrt.(psi))

        R = 5000
        sim = zeros(R,1)

        for l in 1:R
                g = reshape(repeat(randn(n), inner = p),(p, n))'
                sim[l] = n * maximum(2*abs.(mean.(eachcol(tXehattpsi.* g))))
        end

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c .* quantile.(vec(sim), 1 - gamma)

        Ups0 = (1 /sqrt(n)) * sqrt.((x.^2)' * (y.^2))

        lmbda = lmbda0 .* Ups0

    end
    return Dict("lambda0" => lmbda0, "lambda" => lmbda, "Ups0" => Ups0) 
    
end


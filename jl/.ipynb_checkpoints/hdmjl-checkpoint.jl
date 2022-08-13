using Statistics, GLM, DataFrames

# Help functions
function init_values(X, y, number::Int64 = 5, intercept::Bool = true)
    
    if typeof(y) != AbstractVector
        y = vec(y)
    end
    corr = abs.(cor(y, X)[1, :])
    kx = size(X, 2)
    index = sortperm(corr, rev = true)[1: min(number, kx)]
    
    coefficients = zeros(kx)
    
    reg = lm(X[:, index], y)
    coefficients[index] = GLM.coef(reg)
    replace!(coefficients, NaN=>0)
    
    e = y - predict(reg, X[:, index])
    
    res = Dict("coefficients" => coefficients, "residuals" => e)
    
    return res
    #return index
    
end

function lambdaCalculation(; homoskedastic::Bool=false, 
        X_dependent_lambda::Bool=false, 
        lambda_start=nothing, 
        c::Float64=1.1, 
        gamma::Float64=0.1, 
        numSim::Int=5000, 
        y=nothing, 
        x=nothing, 
        par::Bool=true, 
        corecap::Float64=Inf, 
        fix_seed::Bool=true)
    
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

        Ups0 = (1 /sqrt(n)) * sqrt.((y.^2)'*(x.^2))

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
        lmbda0 = c*quantile(vec(sim), 1 - gamma)

        Ups0 = sqrt(var(y, corrected = true))

        # Calculate the final vector of penalty terms
        lmbda = zeros(p,1) .+ lmbda0 * Ups0

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
                sim[l] = n * maximum(2*abs.(mean.(eachcol(tXtpsi.* g))))
        end

        # Initialize lambda based on the simulated quantiles
        lmbda0 = c*quantile(vec(sim), 1 - gamma)

        Ups0 = (1 /sqrt(n)) * sqrt.((y.^2)'*(x.^2))

        lmbda = lmbda0 * Ups0

    end
    return Dict("lambda0" => lmbda0, "lambda" => lmbda, "Ups0" => Ups0) 
    
end

# LassoShooting_fit
function LassoShooting_fit(x, y, lmbda; 
        maxIter::Int = 1000, 
        optTol::Float64 = 10^(-5), 
        zeroThreshold::Float64 = 10^(-6), 
        XX = nothing, Xy = nothing, 
        beta_start = nothing)
        
     """ Shooting LASSO algorithm with variable dependent penalty weights
    Inputs
    x: n by p array, RHS variables
    y: n by 1 array, outcome variable
    lmbda: p by 1 NumPy array, variable dependent penalty terms. The j-th
           element is the penalty term for the j-th RHS variable.
    maxIter: integer, maximum number of shooting LASSO updated
    optTol: scalar, algorithm terminated once the sum of absolute differences
            between the updated and current weights is below optTol
    zeroThreshold: scalar, if any final weights are below zeroThreshold, they
                   will be set to zero instead
    XX: k by k NumPy array, pre-calculated version of x'x
    Xy: k by 1 NumPy array, pre-calculated version of x'y
    beta_start: k by 1 NumPy array, initial weights
    Outputs
    w: k by 1 NumPy array, final weights
    wp: k by m + 1 NumPy array, where m is the number of iterations the
        algorithm took. History of weight updates, starting with the initial
        weights.
    m: integer, number of iterations the algorithm took
    """
    n = size(x)[1]
    p = size(x)[2]
    
    if (isnothing(XX))
        XX = x'*x
    end

    if (isnothing(Xy))
        Xy = x'*y
    end

   if (isnothing(beta_start))
        beta = init_values(x, y, 5, false)["coefficients"]
    else
        beta = beta_start
    end

    wp = beta
    m = 1
    XX2 = XX * 2
    Xy2 = Xy * 2

    while m < maxIter
        beta_old = copy(beta)
        for j in 1:p
            # Compute the Shoot and update variables
            S0 = sum(XX2[j, :] .* beta) - XX2[j, j] .* beta[j] - Xy2[j]
            
            if sum(isnothing.(S0)) >= 1
                beta[j] = 0
            elseif S0 > lmbda[j]
                beta[j] = (lmbda[j] - S0) / XX2[j, j]
            elseif S0 < -lmbda[j]
                beta[j] = (-lmbda[j] - S0) / XX2[j,j]
            elseif abs.(S0) <= lmbda[j]
                beta[j] = 0
            end
        end
        # Update
        wp = hcat(wp, beta)
        # End condition
        if sum(abs.(beta - beta_old)) < optTol
            break
        end
        m = m + 1
    end
    w = beta   

    w[abs.(w) .< zeroThreshold] .= 0
    
    return Dict("coefficients" => w, "coef_list" => wp, "num_it" => m)
    
end


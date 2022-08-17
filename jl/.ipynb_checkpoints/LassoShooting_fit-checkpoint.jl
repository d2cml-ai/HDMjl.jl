function LassoShooting_fit(x, y, lmbda, 
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
function rlassoLATE(x, d, y, z; bootstrap = "none", n_rep = 100, always_takers = true, 
    post = true, intercept = true, never_takers = true)

    x1 = Matrix(x)
    n, p = size(x1)

    function get_mtrx(x1, type = intercept)
        n1, p1 = size(x1)
        if type
            x_mtrx = hcat(ones(n1), x1)
        else
            x_mtrx = x1
        end
        return x_mtrx
    end


    if bootstrap ∉ ["none", "Bayes", "normal", "wild"]
        error("No valid bootstrap")
    end

    lambda = 2.2 * sqrt(n) * quantile(Normal(0.0, 1.0),  1 - (.1 / log(n)) / (2 * (2 * p)) )

    ctrl_num_iter, ctrl_tol = 15, 10^-5

    penalty_homoscedastic = "none"
    penalty_lambda_start = repeat([lambda], p)
    penalty_c = 1.1
    penalty_gamma = 0.1

    indz1, indz0 = [], []

    for i in eachindex(z)
        if z[i] == 1
            append!(indz1, i)
        else
            append!(indz0, i)
        end
    end


    x_z1 = Matrix(x1[indz1, :])
    y_z1 = y[indz1, :]

    x_z0 = Matrix(x1[indz0, :])
    y_z0 = y[indz0, :]

    b_y_z1xL = rlasso(
        x_z1, y_z1, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma, #penalty
        lambda_start = penalty_lambda_start
    )

    ### intercept with main X
    my_z1x = get_mtrx(x1) * b_y_z1xL["coefficients"]

    b_y_z0xL = rlasso(
        x_z0, y_z0, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma, #penalty
        lambda_start = penalty_lambda_start
    )

    my_z0x = get_mtrx(x1) * b_y_z0xL["coefficients"]

    lambda = 2.2 * sqrt(n) * quantile(Normal(0, 1), 1 - (.1 / log(n)) / (2 * (2 * p)))

    if z == d
        md_z1x = ones(n)
        md_z0x = zeros(n)
    else
        x1 = Matrix(x)
        if all([always_takers, never_takers])

            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, lambda = lambda, c = penalty_c, gamma = penalty_gamma)
            yp1 = get_mtrx(x1) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp1)))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, lambda = lambda, c = penalty_c, gamma = penalty_gamma)
            yp0 = get_mtrx(x1) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

        elseif always_takers == false & never_takers == true
            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, lambda = lambda, c = penalty_c, gamma = penalty_gamma)
            yp1 = get_mtrx(x1) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

            md_z0x = zeros(size(x1, 1))

        elseif always_takers == true & never_takers == false 

            md_z1x = ones(size(x, 1))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, lambda = lambda, c = penalty_c, gamma = penalty_gamma)
            yp0 = get_mtrx(x1) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))
        elseif always_takers == false & never_takers == false
            md_z0x = zeros(n)
            md_z1x = ones(n)
        end
    end

    b_z_xl = rlassologit(x1, z, post = post, intercept = intercept)
    yp_b = get_mtrx(x1) * b_z_xl["coefficients"]
    mz_x = 1 ./ (1 .+ exp.(-1 .* (yp_b)))

    mz_x = mz_x .* ((mz_x .> 1e-12) .& (mz_x .< (1 - 1e-12))) .+ (1 - 1e-12) .* (
        mz_x .> 1 .- 1e-12) .+ 1e-12 .* (mz_x .< 1e-12)

    eff = @. (
        z * (y - my_z1x) / mz_x - ((1 - z) * (y - my_z0x)/(1 - mz_x)) + 
        my_z1x - my_z0x) / mean(z * (d - md_z1x)/mz_x - ((1 - z) * (d - md_z0x)/(1 - mz_x)) + 
        md_z1x - md_z0x
        )

    se = sqrt(var(eff)) / sqrt(n)
    late = mean(eff)
    individual = eff

    object = Dict(
        "se" => se, "te"  => late, "individual" => individual, "type" => "LATE",
        "sample_size" => n
    )

    boot = []
    if bootstrap != "none"
        for i in 1:n_rep
            if bootstrap == "Bayes"
                method = Exponential()
                weights = rand(method, n) .- 1
            elseif bootstrap == "normal"
                method = Normal()
                weights = rand(method, n) 
            else
                method = Normal()
                weights = rand(method, n) ./ sqrt(2) .+ (rand(method, n).^2 .- 1) ./ 2
            end
            weights = weights .+ 1

            bt =  mean(@. weights * (z * (y - my_z1x)/mz_x - ((1 - z) * (y - my_z0x)/(1 - mz_x)) + my_z1x - my_z0x))/mean(@. weights * (z * (d - md_z1x)/mz_x - ((1 - z) * (d - md_z0x)/(1 - mz_x)) + md_z1x - md_z0x))
            append!(boot, bt)
        end
        object["boot_se"] = sqrt(var(boot))
        object["type_boot"] = bootstrap
        object["boot_n"] = n_rep
    end
    object

    return object
end

function rlassoATE(x, d, y; bootstrap = "none", n_rep = 500)
    z = copy(d)
    res = rlassoLATE(x, d, y, z, bootstrap = bootstrap, n_rep = n_rep)
    res["type"] = "ATE"
    return res
end

function rlassoLATET(x, d, y, z; bootstrap::String = "none", n_rep::Int64 = 500, post::Bool = true, always_takers::Bool = true, never_takers::Bool = true, intercept::Bool = true)
    n = size(x, 1)
    p = size(x, 2)
    lambda = 2.2 * sqrt(n) * quantile(Normal(0.0, 1.0),  1 - (0.1 / log(n)) / (2 * (2 * p)))
    indz1 = findall(z .== 1)
    indz0 = findall(z .== 0)
    b_y_z0xL = rlasso(x[indz0, :], y[indz0], post = post, intercept = intercept, homoskedastic = "none", c = 1.1, gamma = 0.1, lambda_start = lambda)
    if intercept
        my_z0x = hcat(ones(n), x) * b_y_z0xL["coefficients"]
    elseif !intercept
        my_z0x = x * b_y_z0xL["coefficients"]
    end



    if d == z
        md_z0x = zeros(n)
    else
        if always_takers
            g_d_z0 = rlassologit(x[indz0, :], d[indz0], post = post, intercept = intercept, c = 1.1, gamma = 0.1, lambda = lambda)
            if intercept
                md_z0x = hcat(ones(n), x) * g_d_z0["coefficients"]
            elseif !intercept
                md_z0x = x * g_d_z0["coefficients"]
            end
            md_z0x = @. 1 / (1 + exp(-md_z0x))
        else
            md_z0x = zeros(n)
        end
    end

    b_z_xl = rlassologit(x, z, post = post, intercept = intercept, c = 1.1, gamma = 0.1, lambda = lambda)
    if intercept
        mz_x = hcat(ones(n), x) * b_z_xl["coefficients"]
    elseif !intercept
        mz_x = x * b_z_xl["coefficients"]
    end

    mz_x = @. 1 / (1 + exp(-mz_x))

    mz_x = mz_x .* ((mz_x .> 1e-12) .& (mz_x .< (1 - 1e-12))) .+ (1 - 1e-12) .* (mz_x .> 1 .- 1e-12) .+ 1e-12 .* (mz_x .< 1e-12)    
    
    effnum = (y - my_z0x) - (1 .- z) .* (y - my_z0x)./(1 .- mz_x)
    
    effden = mean((d - md_z0x) - (1 .- z) .* (d - md_z0x)./(1 .- mz_x))

    eff = effnum ./ effden

    se = sqrt(var(eff)) / sqrt(n)
    latet = mean(eff)
    individual = eff
    res = Dict("se" => se, "te" => latet, "individual" => individual, "type" => "LATET", "sample_size" => n)

    if bootstrap != "none"
        boot = fill(NaN, n_rep)
        for i in 1:n_rep
            if bootstrap == "Bayes"
                weights = randexp(n) - 1
            elseif bootstrap == "normal"
                weights = randn(n)
            elseif bootstrap == "wild"
                randn(n) ./ sqrt(2) + (randn(n) .^ 2 .- 1) ./ 2
            end
            weights = weights .+ 1
            boot[i] = mean(weights .* ((y - my_z0x) - (ones(n) - z) .* (y - my_z0x) ./ (ones(n) - mz_x))) / mean(weights .* ((d - md_z0x) - (ones(n) - z) .* (d - md_z0x) ./ (ones(n) - mz_x)))
        end
        res["boot_se"] = sqrt(var(boot))
        res["type_boot"] = bootstrap
        
        return res
    end
    return res
end

function rlassoATET(x, d, y, bootstrap::String = "none", n_rep::Int64 = 500)
    z = copy(d)
    res = rlassoLATET(x, d, y, z, bootstrap = bootstrap, n_rep = n_rep)
    res["type"] = "ATET"
    return res
end
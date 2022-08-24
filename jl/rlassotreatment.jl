using GLM, Statistics
using Distributions, Random

include("hdmjl.jl")
include("rlassologit.jl")

using CSV, DataFrames
xlate = CSV.read("jl/data/xlate.csv", DataFrame)
ylate = CSV.read("jl/data/ylate.csv", DataFrame)
dlate = CSV.read("jl/data/dlate.csv", DataFrame)
zlate = CSV.read("jl/data/zlate.csv", DataFrame)

x = xlate[:, :]
d = dlate[:, 2]
y = ylate[:, 3]
z = zlate[:, 1]

rlassologit(Matrix(x), d)

rlasso(Matrix(x)[:, [3, 4]], x[:, 2])["coefficients"]


### params
#### x, d, y, z

bootstrap = "none"
n_rep = 5 #500
post = true
intercept = true
# always_takers = true
# never_takers = true

function rlassoLATE(x, d, y, z; bootstrap = "none", n_rep = 100, always_takers = true, 
    post = true, intercept = true, never_takers = true)

    ## rlassoLATE with intercept 
    ## bootstrap missing
    # intercept = true


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

    penalty_homoscedastic = false #-> none
    penalty_lambda_start = repeat([lambda], p)
    penalty_c = 1.1
    penalty_gamma = 0.1

    indz1, indz0 = [], []

    for i in eachindex(z)
        # print(i)
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


    # include("help_functions.jl")
    # include("hdmjl.jl")
    # include("LassoShooting_fit.jl")
    
    # init_values(x_z1, y_z1)
    
    b_y_z1xL = rlasso(
        x_z1, y_z1, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma, #penalty
        lambda_start = penalty_lambda_start
        # lambda start not found => via Lambda calculation
    )

    ### intercept with main X
    # b_y_z1xL["coefficients"]
    my_z1x = get_mtrx(x1) * b_y_z1xL["coefficients"]

    b_y_z0xL = rlasso(
        x_z0, y_z0, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma, #penalty
        lambda_start = penalty_lambda_start
        # lambda start not found => via Lambda calculation
    )

    ## with intercept
    my_z0x = get_mtrx(x1) * b_y_z0xL["coefficients"]

    lambda = 2.2 * sqrt(n) * quantile(Normal(0, 1), 1 - (.1 / log(n)) / (2 * (2 * p)))

    # include("rlassologit.jl")

    if z == d
        md_z1x = ones(n)
        md_z0x = zeros(n)
    else
        # intercept true default repeat
        x1 = Matrix(x)
        if all([always_takers, never_takers])

            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp1 = get_mtrx(x1) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp1)))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp0 = get_mtrx(x1) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

        elseif always_takers == false & never_takers == true
            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp1 = get_mtrx(x1) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

            md_z0x = zeros(size(x1, 1))

        elseif always_takers == true & never_takers == false 

            md_z1x = ones(size(x, 1))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp0 = get_mtrx(x1) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))
        elseif always_takers == false & never_takers == false
            md_z0x = zeros(n)
            md_z1x = ones(n)
        end
    end

    ## intercept

    b_z_xl = rlassologit(x1, z, post = post, intercept = intercept)
    yp_b = get_mtrx(x1) * b_z_xl["coefficients"]
    mz_x = 1 ./ (1 .+ exp.(-1 .* (yp_b)))

    mz_x = mz_x .* (mz_x .> 1e-12 .&& mz_x .< (1 - 1e-12)) .+ (1 - 1e-12) .* (
        mz_x .> 1 .- 1e-12) .+ 1e-12 .* (mz_x .< 1e-12)
    
    # @. z * (y - my_z1x) / mz_x - ((1- z) * (y - my_z0x))

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
    # boot

    return object
end

function rlassoATE(x, d, y; bootstrap = "none", n_rep = 500)
    z = copy(d)
    res = rlassoLATE(x, d, y, z, bootstrap = bootstrap, n_rep = n_rep)
    res["type"] = "ATE"
end

rlassoLATE(x, d, y, z, bootstrap = "wild", always_takers = true, never_takers = true, intercept = false, post = false)

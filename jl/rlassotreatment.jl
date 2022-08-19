using GLM
using Distributions

include("rlasso.jl")
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

### params
#### x, d, y, z

# bootstrap = "none"
# # n_rep = 5 #500
# post = true
# intercept = true
# always_takers = true
# never_takers = true

function rlassoLATE(x, d, y, z; boostrap = "none", n_rep = 100, always_takers = true, never_takers = true)

    ## rlassoLATE with intercept 
    ## bootstrap missing
    # intercept = true

    x1 = Matrix(x)
    n, p = size(x1)

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


    x_z1 = Matrix(x[indz1, :])
    y_z1 = y[indz1, :]

    x_z0 = Matrix(x[indz0, :])
    y_z0 = y[indz0, :]


    include("help_functions.jl")
    include("rlasso.jl")
    include("LassoShooting_fit.jl")

    b_y_z1xL = rlasso(
        x_z1, y_z1, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma #penalty
        # lambda start not found => via Lambda calculation
    )

    ### intercept with main X
    my_z1x = hcat(ones(size(x_z1, 1)), x_z1) * b_y_z1xL["coefficients"][:, 2]

    b_y_z0xL = rlasso(
        x_z0, y_z0, post = post, intercept = intercept,
        tol = ctrl_tol, maxIter = ctrl_num_iter, #controls
        homoskedastic = penalty_homoscedastic, c = penalty_c, gamma = penalty_gamma #penalty
        # lambda start not found => via Lambda calculation
    )

    ## with intercept
    my_z0x = hcat(ones(size(x_z0, 1)), x_z0) * b_y_z0xL["coefficients"][:, 2]

    lambda = 2.2 * sqrt(n) * quantile(Normal(0, 1), 1 - (.1 / log(n)) / (2 * (2 * p)))

    include("rlassologit.jl")

    if z == d
        md_z1x = ones(n)
        md_z0x = zeros(n)
    else
        # intercept true default repeat
        x1 = Matrix(x)
        if all([always_takers, never_takers])

            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp1 = hcat(ones(size(x, 1)), Matrix(x)) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp1)))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp0 = hcat(ones(size(x, 1)), Matrix(x)) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

        elseif always_takers == false & never_takers == true
            g_d_z1 = rlassologit(x_z1, d[indz1], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp1 = hcat(ones(size(x1, 1)), Matrix(x)) * g_d_z1["coefficients"]
            md_z1x = 1 ./ (1 .+ exp.(-1 .* (yp0)))

            md_z0x = zeros(size(x1, 1))

        elseif always_takers == true never_takers == false 

            md_z1x = ones(size(x, 1))

            g_d_z0 = rlassologit(x_z0, d[indz0], post = post, intercept = intercept, penalty_lambda = lambda, penalty_c = penalty_c, penalty_gamma = penalty_gamma)
            yp0 = hcat(ones(size(x, 1)), Matrix(x)) * g_d_z0["coefficients"]
            md_z0x = 1 ./ (1 .+ exp.(-1 .* (yp0)))
        elseif always_takers == false & never_takers == false
            md_z0x = zeros(n)
            md_z1x = ones(n)
        end
    end

    ## intercept

    b_z_xl = rlassologit(x, z, post = post, intercept = intercept)
    yp_b = hcat(ones(size(x, 1)), Matrix(x)) * b_z_xl["coefficients"]
    mz_x = 1 ./ (1 .+ exp.(-1 .* (yp_b)))

    mz_x = mz_x .* (mz_x .> 1e-12 .&& mz_x .< (1 - 1e-12)) .+ (1 - 1e-12) .* (
        mz_x .> 1 .- 1e-12) .+ 1e-12 .* (mz_x .< 1e-12)

    eff <- @. (
        z * (y - my_z1x) / mz_x - ((1 - z) * (y - my_z0x)/(1 - mz_x)) + 
        my_z1x - my_z0x) / mean(z * (d - md_z1x)/mz_x - ((1 - z) * (d - md_z0x)/(1 - mz_x)) + 
        md_z1x - md_z0x
        )

    se = sqrt(var(eff)) / sqrt(n)
    late = mean(se)
    individual = eff

    object = Dict(
        "se" => se, "te"  => late, "individual" => individual, "type" => "LATE",
        "sample_size" => n
    )

    return object
end




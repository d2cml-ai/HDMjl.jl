
## params
using GLM, Statistics, CSV, DataFrames

include("hdmjl.jl")

xlate = CSV.read("jl/data/xlate.csv", DataFrame)
dlate = CSV.read("jl/data/dlate.csv", DataFrame)

method = "double selection"

x = xlate[:, [3, 4, 9]]
y = xlate[:, 5]
d = dlate[:, 3]
# CSV.read("jl/data/xlate.csv")

b = [12, 1, 0, 2]

b_bool = []
for i in b
    if i == 0
        append!(b_bool, false)
    else
        append!(b_bool, true)
    end
end
b_bool

function rlassoEffect(
    x, 
    y, 
    d, 
    method = "double selection",
    I3 = nothing,
    post = true
)

    x = Matrix(x)
    y = Matrix(y[:, :])
    d = Matrix(d[:, :])

    n, p = size(x)

    # i3 = Set(I3)

    function as_logical(array)
        b = []
        for i in array
            if i == 0
                append!(b, false)
            else
                append!(b, true)
            end
        end
        return b
    end

    if method == "double selection"
        I1 = rlasso(x, d, post = post)["index"]
        I2 = rlasso(x, y, post = post)["index"]

        if size(i3, 1) == 2
            @. I = I1 + I2 + I3
        else
            @. I =  I1 + I2
        end
        if sum(I) == 0
            I = nothing
        end
    
        inx = []
    
        for i in eachindex(I)
            if I[i] == 1
                append!(inx, i)
            end
        end
    
       
        data = hcat(y, ones(n), d, x[:,inx])
        reg1 = GLM.lm(data[:, Not(1)], data[:, 1])
        alpha = GLM.coef(reg1)[2]
    
        xi = @. GLM.residuals(reg1) * sqrt(n / (n - sum(I) - 1))
    
        if isnothing(I)
            data2 = hcat(d, ones(n))
            reg2 = GLM.lm(data2[:, 1], data2[:, 2])
        else
            reg2 = GLM.lm(data[:, Not(1)], data[: , 1])
        end
    
        v = GLM.residuals(reg2)
    
        var = 1 / n * 1 /mean(v.^2) * mean(v.^2 * xi.^2) * 1 / mean(v.^2)
    
        se = sqrt(var)
    
        tval = alpha / se
    
        # pval = 2 ##3 searching function (`pnorm`)
    
       if isnothing(I)
            no_select = 1
        else
            no_select = 0
        end
    
        res = Dict(
            "epsilon" => xi, "v" => v
        )

        results = Dict(
            "alpha" => alpha, "se" => se, "t" => tval,
            "no_select" => no_select, "coefficients" => alpha, "coefficient" => alpha,
            "coefficients_reg" => GLM.coef(reg1), "selection_index" => I, "residuals" => res,
            "sample_size" => n
        )
    elseif method == "partialling out"
        reg1 = rlasso(x, y, post = post)
        yr = reg1["residuals"]
        reg2 = rlasso(x, d, post = post)
        dr = reg2["residuals"]

        data0 = hcat(y, ones(n), d)
        reg3 = GLM.lm(data0[:, Not(1)], data0[:, 1])
        alpha = GLM.coef(reg3)[2]

        var = vcov(reg3)[2, 2]
        se = sqrt.(var)
        tval = alpha ./ sqrt(var)
        # pval = 
        res = Dict("epsilon" => GLM.residuals(reg3), "v" => dr)

        I1 = reg1["index"]
        I2 = reg2["index"]

        results = Dict(
            "alpha" => alpha, "se" => se, "t" => tval, "coefficients" => alpha,
            "coefficient" => alpha, "coefficients_reg" => GLM.coef(reg1), "selection_index" => I,
            "residuals" => res, "sample_size" => n
        )

    end

    return results


    
end



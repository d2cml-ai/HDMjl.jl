
function rlassoEffect(
        x, 
        y, 
        d;
        method = "double selection",
        I3 = nothing,
        post = true
    )

    #= x = Matrix(x)
    y = Matrix(y[:, :])
    d = Matrix(d[:, :]) =#

    n, p = size(x)

    # i3 = Set(I3)

    if method == "double selection"
        I1 = rlasso(x, d, post = post)["index"]
        I2 = rlasso(x, y, post = post)["index"]

        if !isnothing(I3)
            I = I1 + I2 + I3
        else
            I =  I1 + I2
        end

        if sum(I) == 0
            I = nothing
        end
    
        I[I .> 1] .= 1
        I = BitVector(I)
       
        x = hcat(ones(n), d, x[:, I])
        reg1 = GLM.lm(x, y)
        alpha = GLM.coef(reg1)[2]
    
        xi = GLM.residuals(reg1) .* sqrt(n / (n - sum(I) - 1))
    
        if isnothing(I)
            reg2 = GLM.lm(ones(n, 1), d)
        else
            reg2 = GLM.lm(x[:, Not(2)], reshape(d, n, 1))
        end
    
        v = GLM.residuals(reg2)
    
        var = 1 / n * 1 /mean(v.^2) * mean(v.^2 .* xi.^2) * 1 / mean(v.^2)
    
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

        data0 = hcat(yr, ones(n), dr)
        reg3 = GLM.lm(data0[:, Not(1)], data0[:, 1])
        alpha = GLM.coef(reg3)[2]

        var = vcov(reg3)[2, 2]
        se = sqrt.(var)
        tval = alpha ./ sqrt(var)
        # pval = 
        res = Dict("epsilon" => GLM.residuals(reg3), "v" => dr)

        I1 = reg1["index"]
        I2 = reg2["index"]
        I = as_logical(I1 + I2)
        results = Dict(
            "alpha" => alpha, "se" => se, "t" => tval, "coefficients" => alpha,
            "coefficient" => alpha, "coefficients_reg" => reg1["coefficients"], "selection_index" => I,
            "residuals" => res, "sample_size" => n
        )

    end

    return results


    
end

function rlassoEffects(x, y; index = 1:size(x, 2), I3 = nothing)

    x = Matrix(x)
    y = Matrix(y[:, :])

    if Set(index) > 2
        k = p = length(index)
        # all(all())
    else
        k = p = sum(index)
    end
    n, p0 = size(x, 1)

    coefficients = zeros(k)
    se = zeros(k)
    t = zeros(k)
    lasso_reg = Dict()

    reside = zeros(n, p1)
    residv = zeros(n, p1)

    selection_matrix = zeros(p0, k)
    coef_mat = Dict()
    for i in 1:k
        d = x[:, index[i]]
        xt = x[:, Not(index[i])]
        if isnothing(I3)
            I3m = I3
        else
            I3m = I3[Not(index[i])]
            lasso_reg[i] = try
                rlassoEffect(xt, y, d, method = method, I3 = I3m, post = post)
            catch
                "try-error"
            end
        end
        if lasso_regs[i] == "try-error"
            continue
        else
            coefficients[i] = lasso_regs[i]["alpha"]
            se[i] = lasso_regs[i]["se"]
            t[i] = lasso_regs[i]["t"]
            coef_mat[i] = lasso_reg["coefficients_reg"]
            reside[:, i] = lasso_regs[i]["residuals"]["epsilon"]
            residv[:, i] = lasso_regs[i]["residuals"]["v"]
            selection_matrix[Not(index[i]), i] = lasso_reg["selection_index"]
        end
    end

    residuals = Dict("e" => reside, "v" => residv)
    res = Dict(
        "coefficients" => coefficients, "se" => se, "t" => t, 
        "lasso_regs" => lasso_reg, "index" => index, "sample_size" => n,
        "residuals" => residuals, "coef_mat" => coef_mat, "selection_matrix" => selection_matrix
    )
    return res

    
end

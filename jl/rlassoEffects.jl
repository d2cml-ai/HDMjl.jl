function rlassoEffect(x, y, d; method::String = "double selection", I3::Union{Nothing, BitVector} = nothing, post::Bool = true)
    kx = size(x, 2)
    n = size(x, 1)
    if method == "double selection"
        I1 = rlasso(x, d, post = post)["index"]
        I2 = rlasso(x, y, post = post)["index"]
        
        if isa(I3, BitVector)
            I = I1 + I2 + I3
        else
            I = I1 + I2
        end
        I[I .> 0] .= 1
        I = BitVector(I)
        if sum(I) == 0
            I = nothing
        end
        if !isnothing(I)
            x = hcat(d, x[:, I])
        else
            x = d
        end
        reg1 = GLM.lm(hcat(ones(n), x), y)
        alpha = GLM.coef(reg1)[2]
        xi = GLM.residuals(reg1) * sqrt(n / (n - sum(I) - 1))
        if inothing(I)
            reg2 = GLM.lm(reshape(ones(n), n, 1), d)
        else
            reg2 = GLM.lm(hcat(ones(n), x[:, Not(1)]), d)
        end
        v = GLM.residual(reg2)
        var = (1 / n) * (1 / mean(v .^ 2)) * mean(v .^ 2 .* x .^ 2) * 1 / mean(v .^ 2)
        se = sqrt(var)
        tval = alpha / se
        pval = 2 * cdf(Normal(), -abs(tval))
        if isnothing(I)
            no_selected = 1
        else
            no_selected = 0
        end
        res = Dict("epsilon" => xi, "v" => v)
        results = Dict("alpha" => alpha, "se" => se, "t" => tval, "pval" => pval, 
            "no_selected" => no_selected, "coefficients" => alpha, "coef_reg" => GLM.coef(reg1), 
            "selection_index" => I, "residuals" => res, "sample_size" => n)
    elseif method == "partialling out"
        reg1 = rlasso(x, y, post = post)
        reg2 = rlasso(x, d, post = post)
        yr = reg1["residuals"]
        dr = reg2["residuals"]
        reg3 = GLM.lm(hcat(ones(n), dr), yr)
        alpha = GLM.coef(reg3)[2]
        var = GLM.vcov(reg3)[2, 2]
        se = sqrt(var)
        tval = alpha / se
        pval = 2 * cdf(Normal(), -abs(tval))
        res = Dict("epsilon" => reg3["residuals"], "v" => dr)
        I1 = reg1["index"]
        I2 = reg2["index"]
        I = I1 + I2
        I[I .> 0] .= 1
        I = BitVector(I)
        results = Dict("alpha" => alpha, "se" => se, "t" => tval, "pval" => pval, 
            "no_selected" => no_selected, "coefficients" => alpha, "coef_reg" => GLM.coef(reg1), 
            "selection_index" => I, "residuals" => res, "sample_size" => n)
    end
    return results
end

# function rlassoEffects(x, y; index::Union{BitVector, Vector{Int64}} = Vector(1:size(x, 2)), method::String = "partialling out", I3::Union{Nothing, BitVector} = nothing, post::Bool = true)
#     if isa(index, Vector{Int64})
#         k = p1 = size(index, 1)
#     elseif isa(index, BitVector)
#         k = p1 = sum(index)
#         index = findall(index)
#     end
#     if method == "double selection"
#     n = size(x, 1)
#     coefficients = fill(NaN, k)
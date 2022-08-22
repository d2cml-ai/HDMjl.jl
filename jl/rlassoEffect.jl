
## params

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

    i3 = Set(I3)
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
    
       
        data = hcat(y, d, x[:,inx])
        reg1 = GLM.lm(data[:, Not(1)], data[:, 1])
        alpha = GLM.coef(reg1)[1]
    
        xi = @. GLM.residuals(reg1) * sqrt(n / (n - sum(I) - 1))
    
        if isnothing(I)
            data2 = hcat(d, ones(n))
            reg2 = GLM.lm(data2[:, 1], data2[:, 2])
        else
            reg2 = GLM.lm(data[:, Not(1)], data[: , 1])
        end
    
        v = GLM.residuals(reg2)
    
        var = @. 1 / n * 1 /mean(v^2) * mean(v^2 * xi^2) * 1 / mean(v^2)
    
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

        results = Dict
    end

    
end


Nothing

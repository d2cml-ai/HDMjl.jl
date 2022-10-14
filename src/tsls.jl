function tsls(d, y, z, x::Union{Nothing, Array} = nothing; intercept::Bool = true, homoscedastic::Bool = true) # x::Union{Nothing, DataFrame, Array}
    d = Matrix(d)
    x = Matrix(x)
    z = Matrix(z)
    y = vec(y)
    res = tsls(d, y, z, x, intercept = intercept, homoscedastic = homoscedastic)
    return res
end

function tsls(d::Array, y::Array, z::Array, x::Union{Nothing, Array} = nothing; intercept::Bool = true, homoscedastic::Bool = true) # x::Union{Nothing, DataFrame, Array}
    n = size(y, 1)
    # if intercept == true
    #     d_names = ["d$y" for y = 1:size(d[:,:],2)]
    #     x_names = ["x$y" for y  = 1:size(x,2)];
    #     coef_names = append!(append!(d_names, ["intercept"]), x_names);
    # else 
    #     d_names = ["d$y" for y = 1:size(d[:,:],2)]
    #     x_names = ["x$y" for y  = 1:size(x,2)];
    #     coef_names = append!(d_names,x_names);
    # end
    
    if intercept == true && isnothing(x) == false
        d_names = ["d$y" for y = 1:size(d[:,:],2)]
        x_names = ["x$y" for y  = 1:size(x,2)];
        coef_names = append!(append!(d_names, ["intercept"]), x_names);
        x = hcat(ones(n, 1), x)
    
    elseif intercept == true && isnothing(x) == true
        x = ones(n, 1)
        d_names = ["d$y" for y = 1:size(d[:,:],2)]
        #x_names = ["x$y" for y  = 1:size(x,2)];
        coef_names = append!(d_names, ["intercept"]);
    
    elseif intercept == false && isnothing(x) == true
        d_names = ["d$y" for y = 1:size(d[:,:],2)]
        #x_names = ["x$y" for y  = 1:size(x,2)];
        coef_names = d_names;
    elseif intercept == false && isnothing(x) == false
        d_names = ["d$y" for y = 1:size(d[:,:],2)]
        x_names = ["x$y" for y  = 1:size(x,2)];
        coef_names = append!(d_names, x_names)
    end
    
    a1 = size(d, 2)
    if isnothing(x)
        a2 = 0
    else
        a2 = size(x, 2)
    end
    
    k = a1 + a2
    
    if isnothing(x)
        X = d
        Z = z
    else
        X = hcat(d, x)
        Z = hcat(z, x)
    end
        
    Mxz = X' * Z
    Mzz = pinv(Z' * Z)
    
    M = pinv(Mxz * Mzz * Mxz')
    
    b = M * Mxz * Mzz * (Z' * y)
    
    if homoscedastic
        e = y - X * b
        VC1 = (sum(e .^ 2) / (n - k)) * M
        
    elseif !homoscedastic
        e = y - X * b
        S = 0
        for i in 1:n
            S = S + e[i] ^ 2 * (Z[i, :]' * Z[i, :])
        end
        S = S / n
        VC1 = n .* M * (Mxz * Mzz .* S * Mzz * Mxz') * M
    end
    if isa(VC1, Matrix)
        se = sqrt.(diag(VC1))
    else
        se = sqrt(VC1)
    end
    b = hcat(coef_names, b)
    res = Dict("coefficients" => b, "vcov" => VC1, "se" => se, "residuals" => e, "sample_size" => n)
    return res
end
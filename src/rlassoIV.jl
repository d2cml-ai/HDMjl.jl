function rlassoIV(x, d, y, z; select_Z::Bool = true, select_X::Bool = true, post::Bool = true)
if !select_Z & !select_X
    res = tsls(d, y, z, x, homoskedastic = false)
    #res["sample_size"] = size(x)[1]
    return res
    
elseif select_Z & !select_X
    res = rlassoIVselectZ(x, d, y, z, post = post)
    #res["sample_size"] = size(x)[1]
    return res
    
elseif !select_Z & select_X
    res = rlassoIVselectX(x, d, y, z, post = post)
    #res["sample_size"] = size(x)[1]
    return res
    
elseif select_Z & select_X
    
    Z = hcat(z, x)
    lasso_d_zx = rlasso(Z, d, post = post)
    lasso_y_x = rlasso(x, y, post = post)
    lasso_d_x = rlasso(x, d, post = post)
    if sum(lasso_d_zx["index"]) == 0
        print("No variables in the Lasso regression of d on z and x selected")
        return Dict("alpha" => nan, "se" => nan)
    end
    ind_dzx = lasso_d_zx["index"]
    PZ = d - lasso_d_zx["residuals"]
    lasso_PZ_x = rlasso(x, PZ, post = post)
    ind_PZx = lasso_PZ_x["index"]
    
    if sum(ind_PZx) == 0
        Dr = d .- mean(d)
    else
        Dr = d .- (PZ - lasso_PZ_x["residuals"])
    end
    
    if sum(lasso_y_x["index"]) == 0
        Yr = y .- mean(y)
    else
        Yr = lasso_y_x["residuals"]
    end
    
    if sum(lasso_y_x["index"]) == 0
        Zr = PZ .- mean(x)
    else
        Zr = lasso_PZ_x["residuals"]
    end
    
    res = tsls(Dr, Yr, Zr, intercept = false, homoscedastic = false)
    #res["sample_size"] = size(x)[1]
    return res
end
end


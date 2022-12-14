function rlassoIVselectX(x, d, y, z; post::Bool = true)
    coef_names = try names(d)
    catch
        nothing
    end
    n = size(y, 1)
    numIV = size(z, 2)
    # Z = hcat(z, x)
    lasso_d_x = rlasso(x, d, post = post)
    Dr = lasso_d_x["residuals"]
    lasso_y_x = rlasso(x, y, post = post)
    Yr = lasso_y_x["residuals"]
    Zr = zeros(n, numIV)
    for i in 1:numIV
        lasso_z_x = rlasso(x, z[:, i], post = post)
        Zr[:, i] = lasso_z_x["residuals"]
    end
    
    result = tsls(Dr, Yr, Zr, nothing,intercept = false)
    se = result["se"]
    vcov = result["vcov"]
    if isnothing(coef_names)
        coef = result["coefficients"]
        # coefnames = result["coefnames"]
        res = Dict("coefficients" => coef, "vcov" => vcov, "se" => se, "sample_size" => n)
    else
        coef = result["coefficients"]
        coef[:,1] = coef_names
        res = Dict("coefficients" => coef, "vcov" => vcov, "se" => se, "sample_size" => n)
    end
    return res
end
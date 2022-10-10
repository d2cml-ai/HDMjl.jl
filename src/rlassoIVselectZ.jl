function ginv(X, tol = sqrt(eps(Float64)))
    Xsvd =  svd(X);
    Positive = Xsvd.S .> maximum(tol * Xsvd.S[1]);
    if all(Positive)
        b = Xsvd.V *(1 ./ Xsvd.S .* (Xsvd.U'));
    elseif (!any(Positive))
        b = zeros(size(X)[2],size(X)[1])
    else
        c1 = Xsvd.V[:,Positive]
        c2 = (Xsvd.U[:, Positive])';
        b = c1 * ((1 ./ Xsvd.S[Positive]) .* c2)
    end
end

function rlassoIVselectZ(x, d, y, z; post::Bool = true, intercept::Bool = true)
    
    n = size(y, 1)
    kex = size(x, 2)
    ke = size(d, 2)
    
    d_names = ["d$y" for y = 1:ke]
    x_names = ["x$y" for y  = 1:kex];
    coef_names = append!(d_names,x_names);
    
    Z = hcat(z, x)
    kiv = size(Z, 2)
    select_mat = zeros(0)
    
    # first stage regression
    Dhat = zeros(0)
    flag_const = 0
    for i in 1:ke
        di = d[:, i]
        lasso_fit = rlasso(Z, di, post = post, intercept = intercept)
        if sum(lasso_fit["index"]) == 0
            dihat = zeros(n) .+ mean(di)
            flag_const = flag_const + 1
            if flag_const > 1
                print("No variables selected for two or more instruments leading to multicollinearity problems.")
            end
            if isempty(select_mat)
                select_mat = append!(select_mat, zeros(Bool, kiv))
            else
                select_mat = hcat(select_mat, zeros(Bool, kiv))
            end
        else
            if intercept
                dihat = hcat(ones(n), Z) * lasso_fit["coefficients"]
            else
                dihat = Z * lasso_fit["coefficients"]
            end
            if isempty(select_mat)
                select_mat = append!(select_mat, lasso_fit["index"])
            else
                select_mat = hcat(select_mat, lasso_fit["index"])
            end
        end
        if isempty(Dhat)
            Dhat = append!(Dhat, dihat)
        else
            Dhat = hcat(Dhat, dihat)
        end
    end
    
    Dhat = hcat(Dhat, x)
    d = hcat(d, x)
    
    alpha_hat = ginv(Dhat' * d) * (Dhat' * y)
    residuals = y - d * alpha_hat
    Omega_hat = Dhat' * (Dhat .* (residuals .^ 2))
    Q_hat_inv = pinv(d' * Dhat)
    vcov = Q_hat_inv * Omega_hat * Q_hat_inv'
    alpha_hat = hcat(coef_names, alpha_hat)
    res = Dict("coefficients" => alpha_hat[1:ke, :], "se" => sqrt.(diag(vcov))[1:ke], "residuals" => residuals, "samplesize" => n)
    return res
end
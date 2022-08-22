function rlassoIVselectZ(x, y, d, z; post::Bool = true, intercept::Bool = true)
    
    n = size(y, 1)
    kex = size(x, 2)
    ke = size(d, 2)
    
    Z = hcat(z, x)
    kiv = size(Z, 2)
    select_mat = nothing
    
    # first stage regression
    Dhat = zeros(m, ke)
    flag_const = 0
    for i in 1:ke
        di = d[:, i]
        lasso_fit = rlasso(Z, di, post = post, intercept = intercept)
        if sum(lasso_fir["index"]) == 0
            dihat = zeros(n) +. mean(di)
            flag_const = flag_const + 1
            if flag_const > 1
                print("No variables selected for two or more instruments leading to multicollinearity problems.")
            end
        else
            coef_rlasso = rlasso_fit["coefficients"]

# function rlassoIVselectZ(x, y, d, z)
 
#     d_mtrx = Matrix(d[:, :])
#     x_mtrx = Matrix(x[:, :])
#     z_mtrx = Matrix(z[:, :])
#     y_mtrx = Matrix(y[:, :])

#     n = length(y)

#     kex = size(x_mtrx, 2) #xcolumns
#     ke = size(d_mtrx, 2) #dcolumns

#     ### names columns
#     ###

#     Z = hcat(z_mtrx, x_mtrx) # intercept
#     kiv = size(Z, 2)

#     select_mat = zeros(kiv, ke)
#     Dhat = zeros(n, ke)


#     for i in 1:ke
#         di = d[:, i]
#         lasso_fit = rlasso(Z, di)
#         if false# sum(lasso_fit["index"]) == 0
#             dihat = repeat([mean(di)], n)
#             flag_const = flag_const + 1
#             if flag_const > 1
#                 print("No variables selected for two or more instruments leading to multicollinearity problems.")
#             end
#         else
#             coef_rlasso = lasso_fit["coefficients"]#[:, :x2]
#             dihat = hcat(ones(n), Z) * coef_rlasso # predict
#             # select_mat[:, i] = lasso_fit["index"]
#         end
#         Dhat[:, i] = dihat
#     end

#     Dhat = hcat(Dhat, x_mtrx)
#     d_mtrx = hcat(d_mtrx, x_mtrx)
#     alpha_hat = (y_mtrx' * Dhat) * pinv(d_mtrx' * Dhat)
#     res = y_mtrx .- d_mtrx * alpha_hat'
#     omega_hat = (Dhat .* res.^2)' * Dhat
#     q_hat_inv = pinv(Dhat' * d_mtrx)
#     v_cov = q_hat_inv * omega_hat * q_hat_inv'

#     # -----
#     est = Dict(
#         "coefficients" => alpha_hat[1:ke], 
#         "se" => sqrt.(diag(v_cov))[1:ke],
#         "residuals" => res, 
#         "samplesize" => n
#         )   
#     return est
# end
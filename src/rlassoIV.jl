mutable struct rlassoIV1
    se
    sample_size
    #vcov
    coefficients
    dict
end

function rlassoIV(x, d, y, z; select_Z::Bool = true, select_X::Bool = true, post::Bool = true)
    if select_Z == false && select_X == false
        res = tsls(d, y, z, x, homoscedastic = false)
        #res["coefficients"] = hcat(["d$y" for y = 1:size(d[:,:],2)], res["coefficients"])
        se = res["se"]
        
    elseif select_Z == true && select_X == false
        res = rlassoIVselectZ(x, d, y, z, post = post)
        res["sample_size"] = size(x)[1]
        
    elseif select_Z == false && select_X == true
        res = rlassoIVselectX(x, d, y, z, post = post)
        #res["sample_size"] = size(x)[1]
        
    elseif select_Z == true && select_X == true
        
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
        #res["coefficients"] = hcat(["d$y" for y = 1:size(d[:,:],2)], res["coefficients"])
    end
    se = res["se"]
    sample_size = res["sample_size"]
    #vcov = res["vcov"]
    coefficients = res["coefficients"]
    res1 = rlassoIV1(se, sample_size, coefficients, res);
    return res1;
end

#using Crayons
function r_print(object::rlassoIV1, n_digits = 3)
    if size([object.coefficients])[1] !=  0
        # b = ["X$y" for y = 1:length(object.coefficients)]
        # b = reshape(b,(1,length(b)))
        # a = vcat(b, round.(object.coefficients', digits = digits))
        a = hcat(object.coefficients[:,1], round.(object.coefficients[:,2], digits = n_digits))
        if size(object.coefficients, 1) <= 10
            
            println("Coefficients:\n")
            pretty_table(a[:, 2]', tf = tf_borderless, header = a[:, 1], nosubheader = true, equal_columns_width = true, columns_width = 10, alignment=:c) #, header_crayon =crayon"blue")
        
        elseif string(length(object.coefficients))[count("", string(length(object.coefficients)))-1:count("", string(length(object.coefficients)))-1] == "0" 
            for i in 1:convert(Int, trunc(size(object.coefficients, 1)/10, digits =0))
                pretty_table(a[10*(i-1)+1:10*i, 2]', tf = tf_borderless, header = a[10*(i-1)+1:10*i, 1], nosubheader = true, equal_columns_width = true, columns_width = 10, alignment=:c)#, header_crayon =crayon"green")
                print("\n")
            end
        else
            for i in 1:convert(Int, trunc(size(object.coefficients, 1)/10, digits =0))
                pretty_table(a[10*(i-1)+1:10*i, 2]', tf = tf_borderless, header = a[10*(i-1)+1:10*i, 1], nosubheader = true, equal_columns_width = true, columns_width = 10, alignment=:c)#, header_crayon =crayon"green")
                print("\n")
            end
            pretty_table(a[10*convert(Int, trunc(size(object.coefficients, 1)/10, digits =0))+1:size(object.coefficients, 1), 2]', alignment=:c, nosubheader = true, equal_columns_width = true, columns_width = 10,
                            tf = tf_borderless, header = a[10*convert(Int, trunc(size(object.coefficients, 1)/10, digits =0))+1:size(object.coefficients, 1), 1]) #, header_crayon =crayon"green")
        end
    else 
        print("No coefficients\n")
    end
end

#using Distributions
function r_summary(object::rlassoIV1)
    if size([object.coefficients])[1] != 0
        k = length(object.coefficients[:,2])
        table = zeros(k, 4)
        table[:, 1] .= vec(object.coefficients[:,2])
        table[:, 2] .= object.se
        table[:, 3] .= table[:, 1]./table[:, 2]
        table[:, 4] .= 2 * cdf(Normal(), -abs.(table[:, 3]))
        table1 = DataFrame(hcat(object.coefficients[:,1], table), :auto)
        table1 = rename(table1, [" ", "coeff.", "se.", "t-value", "p-value"])
        print("Estimates and Significance Testing of the effect of target variables in the IV regression model", 
                "\n")
        pretty_table(table1, show_row_number = false, header = [" ", "coeff.", "se.", "t-value", "p-value"], tf = tf_borderless)
        print("---", "\n", "Signif. codes:","\n", "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("\n")
        return table1;
    else
        print("No coefficients\n")
        #table = []
    end
    #return table;;
end

function r_confint(object::rlassoIV1, level = 0.95)
    n = object.sample_size
    k = length(object.coefficients)
    cf = object.coefficients[:,2:end]
    #pnames <- names(cf)
    # if (missing(parm)) 
    #     parm <- pnames else if (is.numeric(parm)) 
    #       parm <- pnames[parm]
    a = (1 - level)/2
    a = [a, 1 - a]
    fac = quantile.(Normal(), a)
    pct = string.(round.(a; digits = 3)*100, "%")
    ses = object.se
    c_i = []
    for i in 1:length(cf)
        if i == 1
            c_i = (cf[i] .+ ses[i] .* fac)[:,:]'
        else
            c_i = vcat(c_i, (cf[i] .+ ses[i] .* fac)[:,:]')
        end
    end
    table1 = DataFrame(hcat(object.coefficients[:,1],c_i), :auto)
    table1 = rename(table1, append!([" "],pct))
    ci = pretty_table(table1; header = append!([" "],pct), show_row_number = false, tf = tf_borderless)
    return table1;
end
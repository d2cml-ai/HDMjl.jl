function data_formula(frml::FormulaTerm, data::DataFrame)
    form = apply_schema(frml, schema(frml, data))
    res, pred = modelcols(form, data)

    coef_names = coefnames(form.rhs)

    print(coef_names, "\n")
    if length(coef_names) < 2
        x_df = pred
    else
        x_df = DataFrame(pred[:, :], coef_names)
    end
    return (res, x_df)
    # return coef_names
end
# using StatsModels, StatsBase, DataFrames, StableRNGs; rng = StableRNG(1);


# df = DataFrame(y = rand(rng, 9), a = float(1:9), b = rand(rng, 9), c = categorical(repeat(["a","b","c"], 3)))




# function data_formula(frml::FormulaTerm, data::DataFrame)
#     form = apply_schema(frml, schema(data))
#     df = DataFrame(modelcols(form.rhs, data), coefnames(form.rhs))
#     return df
# end

# func

# f = @formula(y ~ poly(b + a, 5) * c + a)

# data_formula(f, df)



# poly(x, n) = x^n

# f = @formula(y ~ 1+  (a + b + c) * (a + b + c))

# apply_schema(f, schema(df))

# df

# y, x = data_formula(f, df)

# using CSV, DataFrames, StatsModels, StatsBase

# data = CSV.read(download("https://gist.githubusercontent.com/TJhon/158daa0c2dd06010d01a72dae2af8314/raw/61df065c98ec90b9ea3b8598d1996fb5371a64aa/rnd.csv"), DataFrame) 

# ModelMatrix(ModelFrame(@formula(y ~ (x1 + x2 + x3 + x4 + x5)^2), data)).m



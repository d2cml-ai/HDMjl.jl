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

function data_formula(x_names::Array{String} = nothing, y_name::String = nothing, Data = nothing; upto::Int = 2, intercept::Bool = false)
    # ref: https://stackoverflow.com/questions/74117969/statsmodels-jl-exponential-formulas-formulay-x1-x2-x3-x4-x52-in/74118944#74118944
    """
    return a dataframe of covariates
  
    - `R`: model.matrix(y ~ (x1 + x3 + s9 + exp3)^2, data = df)
    - `Julia`: data_formula(["x1", "x2", "s9", "exp3"], y_name = "y", data = df, intercept = false, upto = 2)
  
    """
  
    xs = term.(Symbol.(x_names))
  
    subsets(X; from=0, upto=upto) =
    Iterators.flatten(combinations(X,i) for i=max(0,from):min(upto,length(X)))
  
  
    term_vec = collect(subsets(xs; from = 1, upto = upto))
  
    rhs = map(x -> reduce(&, x), term_vec)
    rhs_names = vcat([join(string.(x),'*') for x in term_vec])
  
    x1 = ModelMatrix(ModelFrame(FormulaTerm(Term(Symbol(y_name)),Tuple(rhs)), Data)).m
  
    if intercept
      x1 = x1
      rhs_names = vcat("(Intercept)", rhs_names)
    else
      x1 = x1[:, Not(1)]
      rhs_names = rhs_names
    end
    
    x_df = DataFrame(x1, rhs_names)
  
    return x_df
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



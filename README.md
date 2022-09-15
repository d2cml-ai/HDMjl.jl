# HDMjl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://d2cml-ai.github.io/HDMjl.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://d2cml-ai.github.io/HDMjl.jl/dev/)
[![Build Status](https://github.com/d2cml-ai/HDMjl.jl/workflows/CI/badge.svg)](https://github.com/d2cml-ai/HDMjl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/d2cml-ai/HDMjl.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/d2cml-ai/HDMjl.jl)

### HDMjl.jl
+ This package is a port of the `hdm` library in R. A collection of methods for estimation and quantification of uncertainty in high-dimensional approximately sparse models. Based on Chernozukov, Hansen and Spindler (2016).

### Getting started

To install the stable version of the package, you may acquire the package from the Julia General Registry by using

```julia
julia> ] add HDMjl
```

or

```julia
julia> import Pkg; Pkg.add("HDMjl")
```

You may also install the dev version of the package by directly acquiring it from the [repository](https://github.com/d2cml-ai/HDMjl.jl) by using

```julia
julia> ] add https://github.com/d2cml-ai/HDMjl.jl
```

or 

```julia
julia> import Pkg; Pkg.add(url = "https://github.com/d2cml-ai/HDMjl.jl")
```

If the compatibility conditions are met, the package should install automatically, and you may load the package:

```julia
julia> using HDMjl
```

### Prediction using Lasso and Post-Lasso

```julia
julia> using Random

julia> Random.seed!(1234);

julia> n = 100;

julia> p = 100;

julia> s = 3;

julia> X = randn(n, p);

julia> beta = vcat(fill(5, s), zeros(p - s));

julia> Y = X * beta + randn(n);
```

The Post-Lasso procedure fits an OLS regression excluding the variables not previously selected by Lasso. The `rlasso` algorithm uses the standard errors of the residuals from this regression to evaluate whether there has been a gain in the goodness of the fit in the current iteration. Just like most of the functions in the package, `rlasso` returns a dictionary with the results of the regression.

We can estimate the models using Lasso

```julia
julia> rlasso(X, Y, post = false)
Dict{String, Any} with 15 entries:
  "tss"          => 6512.49
  "dev"          => [6.75884, -13.5819, -2.81122, -3.94462, 17.3342, -1.2805, 3.16503, -4.74853, 6.944, 15.2907  …  …
  "model"        => [0.390896 0.179228 … 2.36678 2.01764; -0.720606 -1.12332 … 0.169248 -0.831435; … ; 1.2457 0.7669…
  "loadings"     => [1.70326, 1.86338, 2.02143, 1.85829, 1.5416, 1.74625, 1.94735, 1.38887, 1.7228, 1.59366  …  1.65…
  "sigma"        => 1.71111
  "lambda0"      => 81.3601
  "lambda"       => [138.577, 151.605, 164.464, 151.191, 125.424, 142.075, 158.436, 112.998, 140.167, 129.66  …  134…
  "intercept"    => -0.118988
  "iter"         => 16
  "residuals"    => [1.8377, -2.33523, 0.707157, -0.0587436, 3.81226, 0.637385, 0.117754, -0.209206, 1.49168, 2.2032…
  "rss"          => 289.863
  "index"        => Bool[1, 1, 1, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "beta"         => [4.15731, 4.35612, 3.69875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, …
  "options"      => Dict{String, Any}("intercept"=>true, "post"=>false, "meanx"=>[-0.217494 0.000263084 … -0.0073734…
  "coefficients" => [-0.118988, 4.15731, 4.35612, 3.69875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0,…
```
and Post-Lasso

```julia
julia> rlasso(X, Y, post = true)
Dict{String, Any} with 15 entries:
  "tss"          => 6512.49
  "dev"          => [6.75884, -13.5819, -2.81122, -3.94462, 17.3342, -1.2805, 3.16503, -4.74853, 6.944, 15.2907  …  …
  "model"        => [0.390896 0.179228 … 2.36678 2.01764; -0.720606 -1.12332 … 0.169248 -0.831435; … ; 1.2457 0.7669…
  "loadings"     => [0.93007, 0.992403, 0.863634, 1.00966, 0.876833, 0.858748, 1.00182, 0.892263, 1.07537, 1.01695  …
  "sigma"        => 0.925277
  "lambda0"      => 81.3601
  "lambda"       => [75.6706, 80.7419, 70.2653, 82.1458, 71.3392, 69.8678, 81.5081, 72.5946, 87.4919, 82.7389  …  68…
  "intercept"    => 0.0258985
  "iter"         => 5
  "residuals"    => [0.733002, 0.22571, 1.06845, 1.34666, 0.818648, 0.575327, -0.519747, 0.985208, -0.000283277, -0.…
  "rss"          => 84.7576
  "index"        => Bool[1, 1, 1, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "beta"         => [4.94557, 5.14366, 4.8095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0…
  "options"      => Dict{String, Any}("intercept"=>true, "post"=>true, "meanx"=>[-0.217494 0.000263084 … -0.00737349…
  "coefficients" => [0.0258985, 4.94557, 5.14366, 4.8095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, …
```

### Inference on Target Coefficients through Orthogonal Estimating Equations

Following Chernozhukov, Hansen and Spindler (2015), the `HDMjl` package makes use of orthogonal estimating equations methods to reduce estimation bias. We can do this through the `rlassoEffect` function which does orthogonal estimation using `double selection` by default.

We can use this method for the Barro & Lee (1994) dataset, which has a large amount of covariates (61) relative to the sample size (90). Selecting covariates through Post-Lasso gives us more precise estimators.

```julia
julia> using CodecXz

julia> using RData

julia> using DataFrames

julia> url = "https://github.com/cran/hdm/raw/master/data/GrowthData.rda";

julia> GrowthData = load(download(url))["GrowthData"];

julia> y = GrowthData[:, 1];

julia> d = GrowthData[:, 3];

julia> X = Matrix(GrowthData[:, Not(1, 2, 3)]);

julia> rlassoEffect(X, y, d, method = "double selection")
Dict{String, Any} with 10 entries:
  "alpha"            => -0.0500059
  "t"                => -3.16666
  "se"               => 0.0157914
  "no_select"        => 0
  "coefficients_reg" => [-0.406451, -0.0500059, -0.0782423, -0.574676, 0.0511529, -0.0470218, 0.212279, -0.000376038, 0…
  "sample_size"      => 90
  "coefficient"      => -0.0500059
  "selection_index"  => Bool[1, 0, 1, 0, 1, 0, 0, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  "residuals"        => Dict("v"=>[0.497555, 0.183798, 0.0705184, -0.123959, 0.0872214, 0.311811, 0.273583, 0.800463, -…
  "coefficients"     => -0.0500059
```

We can also use `partialling out` for the orthogonal estimating equations.

```julia
julia> rlassoEffect(X, y, d, method = "partialling out")
Dict{String, Any} with 9 entries:
  "alpha"            => -0.0498115
  "t"                => -3.57421
  "se"               => 0.0139364
  "coefficients_reg" => [0.0581009, -0.0755655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0…
  "sample_size"      => 90
  "coefficient"      => -0.0498115
  "selection_index"  => Any[true, false, true, false, true, false, false, false, false, false  …  false, false, false, …
  "residuals"        => Dict("v"=>[0.522248, 0.130278, 0.072321, -0.131969, 0.0984047, 0.357306, 0.294098, 0.797784, -0…
  "coefficients"     => -0.0498115
```

### Instrumental Variable Estimation in High-Dimentional Settings

The `rlassoIV` function is able to select exogenous variables (`X_select = true`), instrumental variables (`Z_select = true`) by default, and use orthogonal estimating ecuations through partialling out for a two-stage least squares regression. We also supply the `tsls` function, which computes two-stage least squares estimates.

We desmonstrate this with the eminent domain data used by Belloni, Chen, Chernozhukov & Hansen (2012):

```julia
julia> using Statistics

julia> url = "https://github.com/cran/hdm/raw/master/data/EminentDomain.rda";

julia> EminentDomain = load(download(url))["EminentDomain"];

julia> z = EminentDomain["logGDP"]["z"];

julia> x = EminentDomain["logGDP"]["x"];

julia> d = EminentDomain["logGDP"]["d"];

julia> y = EminentDomain["logGDP"]["y"];

julia> x = x[:, (mean(x, dims = 1) .> 0.05)'];

julia> z = z[:, (mean(z, dims = 1) .> 0.05)'];

julia> rlassoIV(x, d, y, z)
Dict{String, Any} with 5 entries:
  "se"           => [0.128507]
  "sample_size"  => 312
  "vcov"         => [0.0165139;;]
  "residuals"    => [-0.20468; 0.0311701; … ; 0.252309; 0.335146;;]
  "coefficients" => [-0.0238347;;]
```

### Inference on Treatment Effects on a High-Dimensional Setting

The `rlassoATE`, `rlassoLATE`, `rlassoATET`, and `rlassoLATET` functions are included for estimation of the corresponding treatment effect statistics.

```julia
julia> url = "https://github.com/cran/hdm/raw/master/data/pension.rda";

julia> pension = load(download(url))["pension"];

julia> y = pension[:, "tw"];

julia> d = pension[:, "p401"];

julia> z = pension[:, "e401"];

julia> X = Matrix(pension[:, ["i2", "i3", "i4", "i5", "i6", "i7", "a2", "a3", "a4", "a5", "fsize", "hs", "smcol", "col", "marr", "twoearn", "db", "pira", "hown"]]);

julia> rlassoATE(X, d, y)
Dict{String, Any} with 5 entries:
  "se"          => 1930.68
  "individual"  => [-30618.3, -57537.6, -71442.9, 21383.3, -2.32925e5, 3.40765e5, 97143.9, -286.995, 21439.9, 99072.0  …
  "sample_size" => 9915
  "te"          => 10180.1
  "type"        => "ATE"

julia> rlassoATET(X, d, y)
Dict{String, Any} with 6 entries:
  "se"          => 2944.43
  "individual"  => [-21536.4, -52877.2, -1.44867e5, -2739.29, -307741.0, 7.3912e5, 1.73107e5, 12929.3, -2569.57, 62331.…
  "sample_size" => 9915
  "te"          => 12628.5
  "type"        => "ATET"

julia> rlassoLATE(X, d, y, z)
Dict{String, Any} with 5 entries:
  "se"          => 2326.9
  "individual"  => [-50526.8, -1.39158e5, -1.37102e5, 38508.0, -6.5644e5, 7.94317e5, 2.50222e5, 71721.0, 39272.5, 1.440…
  "sample_size" => 9915
  "te"          => 12992.1
  "type"        => "LATE"

julia> rlassoLATET(X, d, y, z)
Dict{String, Any} with 6 entries:
  "se"          => 3645.28
  "individual"  => [-35580.5, -90558.0, -1.83628e5, -5303.13, -8.0766e5, 1.88668e6, 4.94743e5, 18436.0, -4847.72, 74008…
  "sample_size" => 9915
  "te"          => 15323.2
  "type"        => "LATET"

```


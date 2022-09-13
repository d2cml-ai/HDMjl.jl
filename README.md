# HDMjl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://d2cmjl-ai.github.io/HDMjl.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://d2cmjl-ai.github.io/HDMjl.jl/dev/)
[![Build Status](https://github.com/d2cmjl-ai/HDMjl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/d2cmjl-ai/HDMjl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/d2cmjl-ai/HDMjl.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/d2cmjl-ai/HDMjl.jl)

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

```julia
julia> Random.seed!(1234);

julia> n = 5000;

julia> p = 20;

julia> X = randn(n, p - 1);

julia> beta = ones(p - 1);

julia> d = randn(n);

julia> alpha = 1.0;

julia> Y = alpha * d + X * beta + randn(n);

```

```julia
julia> rlassoEffect(X, Y, d, method = "double selection")
Dict{String, Any} with 10 entries:
  "alpha"            => 0.995604
  "t"                => 72.0529
  "se"               => 0.0138177
  "no_select"        => 0
  "coefficients_reg" => [0.00381425, 0.995604, 0.992196, 1.00151, 0.973211, 0.982338, 1.01797, 0.978455, 1.01751, 1.003…
  "sample_size"      => 5000
  "coefficient"      => 0.995604
  "selection_index"  => Bool[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  "residuals"        => Dict("v"=>[-1.47981, -1.77528, -2.14908, -0.628691, -0.209521, -0.577875, -0.634301, -0.0914649…
  "coefficients"     => 0.995604
```

We can also use the `partialling out` method:

```julia
julia> rlassoEffect(X, Y, d, method = "partialling out")
Dict{String, Any} with 9 entries:
  "alpha"            => 0.99166
  "t"                => 71.0117
  "se"               => 0.0139648
  "coefficients_reg" => [0.014376, 1.00644, 1.01451, 0.972988, 0.980083, 1.01576, 0.971863, 0.973955, 0.992963, 0.97425…
  "sample_size"      => 5000
  "coefficient"      => 0.99166
  "selection_index"  => Any[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, t…
  "residuals"        => Dict("v"=>[-1.43987, -1.73542, -2.00637, -0.544431, -0.111036, -0.601902, -0.667784, -0.0965776…
  "coefficients"     => 0.99166
```

#### Estimating treatment effect in a linear model with many confounding factors

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

# HDMjl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://d2cml-ai.github.io/HDMjl.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://d2cml-ai.github.io/HDMjl.jl/dev/)
[![Build Status](https://github.com/d2cml-ai/HDMjl.jl/workflows/CI/badge.svg)](https://github.com/d2cml-ai/HDMjl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/d2cml-ai/HDMjl.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/d2cml-ai/HDMjl.jl)

### HDMjl.jl
+ This package is a port of the `hdm` library in R. A collection of methods for estimation and quantification of uncertainty in high-dimensional approximately sparse models, based on: 

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., & Newey, W. (2017). Double/debiased/neyman machine learning of treatment effects. American Economic Review, 107(5), 261-65.

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
julia> lasso_reg = rlasso(X, Y, post = false)

julia> r_summary(lasso_reg)
    Post-Lasso Estimation: false
    Total number of variables: 100
    Number of selected variables: 9
    ---
     
============ ==============
  Variable    Estimate    
============ ==============
  Intercept   -0.0588327
  V 1         4.84428
  V 2         4.73331
  V 3         4.99116
  V 4         -0.0166025
  V 43        -0.10963
  V 64        0.000400857
  V 69        -0.0359718
  V 94        0.00666321
  V 100       0.166262
============ ==============

    ----
    Multiple R-squared: 0.9883821717933302
    Adjusted R-squared: 0.9872203889726632
```
and Post-Lasso

```julia
julia> post_lasso_reg = rlasso(X, Y, post = true)

julia> r_summary(post_lasso_reg)
    Post-Lasso Estimation: true
    Total number of variables: 100
    Number of selected variables: 3
    ---
     
============ ==============
  Variable    Estimate    
============ ==============
  Intercept   -0.00682754
  V 1         5.00958
  V 2         4.93178
  V 3         5.17705
============ ==============

    ----
    Multiple R-squared: 0.9878595381779292
    Adjusted R-squared: 0.9874801487459894

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

julia> doublesel_effect = rlassoEffect(X, y, d, method = "double selection");

julia> r_summary(doublesel_effect);
Estimates and significance testing of the effect of target variables
  Row   Estimate.   Std. Error    t value     Pr(>|t|) 

    1    -0.05001      0.01579   -3.16719   0.00154 **
---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We can also use `partialling out` for the orthogonal estimating equations.

```julia
julia> lasso_effect   = rlassoEffect(X, y, d, method = "partialling out")

julia> r_summary(lasso_effect);
Estimates and significance testing of the effect of target variables
  Row   Estimate.   Std. Error    t value      Pr(>|t|) 

    1    -0.04981      0.01394   -3.57317   0.00035 ***
---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
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

julia> lasso_IV_XZ  = rlassoIV(x, d, y, z)

julia> r_summary(lasso_IV_XZ);
Estimates and Significance Testing of the effect of target variables in the IV regression model
         coeff.       se.    t-value    p-value 

  d1   -0.02383   0.12851   -0.18543   0.85289
---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
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

julia> pension_ate  = rlassoATE(X, d, y)

julia> r_summary(pension_ate);
    ------
    Post-Lasso estimation: true
    Intercept: true
    Control: 0
    Total number of variables: 19
    Number of selected variables: 9 
    ------
    
 
============ ============
  Variable    Estimate  
============ ============
  Intercept   -2.07033
  V 1         -0.237913
  V 3         0.618819
  V 4         0.846136
  V 5         1.10569
  V 6         1.34217
  V 10        -0.33151
  V 16        0.0382348
  V 17        0.620232
  V 18        0.335563
============ ============
rlassologit
...
  Coeff     SE        t.value 
========== ========= ==========
  10180.1   1930.68   5.2728
========== ========= ==========

julia> pension_atet  = rlassoATET(X, d, y)

julia> r_summary(pension_atet);
    ------
    Post-Lasso estimation: true
    Intercept: true
    Control: 0
    Total number of variables: 19
    Number of selected variables: 6 
    ------
    
 
============ ============
  Variable    Estimate  
============ ============
  Intercept   -1.79587
  V 1         -0.608675
  V 5         0.622942
  V 6         0.839653
  V 16        0.199394
  V 17        0.643286
  V 18        0.374925
============ ============
rlassologit
    Estimation and significance tesing of the treatment effect
    Type: ATET
    Bootstrap: none
...
  Coeff     SE        t.value 
========== ========= ==========
  12628.5   2944.43   4.28893
========== ========= ==========

julia> pension_late = rlassoLATE(X, d, y, z)

julia> r_summary(pension_late);
    ------
    Post-Lasso estimation: true
    Intercept: true
    Control: 0
    Total number of variables: 19
    Number of selected variables: 10 
    ------
    
 
============ ============
  Variable    Estimate  
============ ============
  Intercept   -1.58403
  V 1         -0.329602
  V 3         0.657641
  V 4         0.836492
  V 5         1.11528
  V 6         1.21348
  V 8         0.142622
  V 10        -0.299557
  V 16        0.0516196
  V 17        1.03219
  V 18        0.135758
============ ============
rlassologit
    Estimation and significance tesing of the treatment effect
    Type: LATE
    Bootstrap: none
    
========== ======== ==========
  Coeff     SE       t.value 
========== ======== ==========
  12992.1   2326.9   5.58344
========== ======== ==========

julia> pension_latet  = rlassoLATET(X, d, y, z)

julia> r_summary(pension_latet);
    ------
    Post-Lasso estimation: true
    Intercept: true
    Control: 0
    Total number of variables: 19
    Number of selected variables: 5 
    ------
    
 
============ ============
  Variable    Estimate  
============ ============
  Intercept   -1.25636
  V 1         -0.714199
  V 5         0.677564
  V 6         0.794049
  V 16        0.212127
  V 17        1.05388
============ ============
rlassologit
    Estimation and significance tesing of the treatment effect
    Type: LATET
    Bootstrap: none
    
========== ========= ==========
  Coeff     SE        t.value 
========== ========= ==========
  15323.2   3645.28   4.20357

```


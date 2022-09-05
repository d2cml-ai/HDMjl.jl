# HDMjl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://d2cmjl-ai.github.io/HDMjl.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://d2cmjl-ai.github.io/HDMjl.jl/dev/)
[![Build Status](https://github.com/d2cmjl-ai/HDMjl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/d2cmjl-ai/HDMjl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/d2cmjl-ai/HDMjl.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/d2cmjl-ai/HDMjl.jl)

### HDMjl.jl
+ A collection of methods for estimation and quantification of uncertainty in high-dimensional approximately sparse models. Based on Chernozukov, Hansen and Spindler (2016).

### Getting started

To install the stable version of the package, you may acquire the package from the Julia General Registry by using

```julia-repl
] add HDMjl
```

in the REPL, or

```julia
import Pkg; Pkg.add("HDMjl")
```

You may also install the dev version of the package by directly acquiring it from the [repository](https://github.com/d2cml-ai/HDMjl.jl) by using

```julia-repl
] add https://github.com/d2cml-ai/HDMjl.jl
```

in the REPL, or 

```julia
import Pkg; Pkg.add(url = "https://github.com/d2cml-ai/HDMjl.jl")
```

If the compatibility conditions are met, the package should install automatically, and you may load the package:

```julia
using HDMjl
```

### Prediction using Lasso and Post-Lasso

```julia
Random.seed!(1234)
n = 100
p = 100
s = 3
X = randn(n, p)
beta = vcat(fill(5, s), zeros(p - s))
Y = X * beta + randn(n)
```

We estimate the models using Lasso

```julia
lasso = rlasso(X, Y, post = false)
```
 and Post-Lasso

```julia
lasso = rlasso(X, Y, post = true)
```
# using Statistics, GLM, DataFrames, LinearAlgebra, GLMNet, Random

module HDMJL

export lambdaCalculation, init_values, as_logical, LassoShooting_fit, rlasso, rlassoEffect, rlassoEffects, rlassoIVselectX, rlassoIVselectZ, rlassoIV, rlassologit, rlassologitEffect, rlassologitEffects, rlassoATE, rlassoATET, rlassoLATE, rlassoLATET, tsls

include("help_functions.jl")
include("LassoShooting_fit.jl")
include("rlasso.jl")
include("rlassoEffect.jl")
include("rlassoIVselectX.jl")
include("rlassoIVselectZ.jl")
include("rlassoIV.jl")
include("rlassologit.jl")
include("rlassotreatment.jl")
include("tsls.jl")

end
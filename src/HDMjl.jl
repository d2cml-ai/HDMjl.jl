module HDMjl

export lambdaCalculation, init_values, as_logical, LassoShooting_fit, rlasso, rlassoEffect, rlassoEffects, rlassoIVselectX, rlassoIVselectZ, rlassoIV, rlassologit, rlassologitEffect, rlassologitEffects, rlassoATE, rlassoATET, rlassoLATE, rlassoLATET, tsls, r_summary, r_print, r_confint, r_predict, get_data, data_formula

using Statistics, GLM, DataFrames, LinearAlgebra, GLMNet, Random, PrettyTables, Distributions, CSV, RData, CodecXz, StatsModels, StatsBase

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
include("formula.jl")

end

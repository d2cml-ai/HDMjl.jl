using CSV, DataFrames, GLM, GLMNet

ajr = CSV.read("Data/")

y = ajr.GDP
x_form = @formula(GDP ~ Latitude + Latitude2 + Africa + Asia + Namer + Samer)
x = modelmatrix(x_form, ajr)
d = ajr.Exprop
z = ajr.logMort
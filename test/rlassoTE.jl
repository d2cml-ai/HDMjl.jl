using RData, DataFrames, CodecXz

url = "https://github.com/cran/hdm/raw/master/data/pension.rda";
pension = load(download(url))["pension"];

y = pension[:, "tw"];
d = pension[:, "p401"];
z = pension[:, "e401"];
X = Matrix(pension[:, ["i2", "i3", "i4", "i5", "i6", "i7", "a2", "a3", "a4", "a5", "fsize", "hs", "smcol", "col", "marr", "twoearn", "db", "pira", "hown"]]);


include("../src/HDMjl.jl")
x
ss = HDMjl.rlassoATET(X, d, y);
HDMjl.r_summary(ss);
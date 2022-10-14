### 3.2
# pwd()

reload() = include("../src/HDMjl.jl")

# reload()

using CSV, DataFrames, PrettyTables

function r_data(n = 1)
    n_m = "data/" * string(n) * ".csv"
    print(n_m )
    dta = CSV.read(n_m, DataFrame)
    return dta
end

function intercept(X)
    n, p = size(X)
    X1 = hcat(ones(n), X, makeunique = true)
    return Matrix(X1)
end

using CSV
url = "https://raw.githubusercontent.com/d2cml-ai/HDMjl.jl/prueba/data/3_2.csv"
dta = DataFrame(CSV.read(download(url), DataFrame));
n, p = size(dta);
Y = dta[:,1];
X = Matrix(dta[:,2:end]);

reload()
lasso_reg = hdm.rlasso(X, Y, post = false);
hdm.r_summary(lasso_reg)


r_32 = r_data("3.2")

y = r_32[:, 1]
x = r_32[:, Not(1)]
# intercept(x)

reload()
lasso_reg = hdm.rlasso(x, y, post = false)
hdm.r_summary(lasso_reg)

yhat_lasso = hdm.r_predict(lasso_reg)
d_new = r_data("3.2_new")
Xnew = d_new[:, Not(1)]
ynew = d_new[:, 1]
lasso_reg["coefficients"]

yhat_lasso_new = hdm.r_predict(lasso_reg, xnew = Matrix(Xnew))
post_lasso_reg = hdm.rlasso(x, y, post = true)
y_hat_postlasso = hdm.r_predict(post_lasso_reg, xnew = Matrix(Xnew))
## TODO: Implementar la funcion print para rlasso
hdm.r_summary(post_lasso_reg)

using Statistics

y_hat_postlasso
# ynew
mean(abs.(ynew[:, 1] - yhat_lasso_new)), mean(abs.(ynew[:, 1] - y_hat_postlasso))

################# 4
### 4.1

r_41 = r_data(4.1)

x_41 = r_41[:, Not(1)]
y_41 = r_41[:, 1]

using GLM

full_fit = GLM.lm(intercept(x_41), y_41)

est = round(coeftable(full_fit).cols[1][2], digits = 5)
s_td = round(coeftable(full_fit).cols[2][2], digits = 5)
print("Estimate: $est ($s_td)")

d_41 = x_41[:, 1]
X1 = x_41[:, Not(1)] 

lm_y = lm(intercept(X1), y_41)
lm_d = lm(intercept(X1), d_41)
# lm_y
n = size(r_41, 1)
rY = GLM.residuals(lm_y)
rd = GLM.residuals(lm_d)
partial_fit_ls = lm(hcat(ones(n), rd), rY)
est = round(coeftable(partial_fit_ls).cols[1][2], digits = 5)
s_td = round(coeftable(partial_fit_ls).cols[2][2], digits = 5)
print("Estimate: $est ($s_td)")


rY = hdm.rlasso(X1, y_41)["residuals"]
rd = hdm.rlasso(X1, d_41)["residuals"]
# intercept(rd)
# rY

partial_fit_ls = GLM.lm(hcat(ones(n), rd), rY[:, 1])
est = round(coeftable(partial_fit_ls).cols[1][2], digits = 6)
s_td = round(coeftable(partial_fit_ls).cols[2][2], digits = 6)
print("Estimate: $est ($s_td)")

print("\n\n\n")

### rlassoEffect

Eff = hdm.rlassoEffect(x_41[:, Not(1)], y_41, x_41[:, 1], method = "partialling out");
hdm.r_summary(Eff);
reload()
x[:, [1, 2]]


reload()
Eff = hdm.rlassoEffect(x_41[:, Not(1)], y_41, x_41[:, 1], method = "double selection");
hdm.r_summary(Eff);



########## 4.2

r_42 = r_data(4.2)
x_42 = r_42[:, Not(1)]
y_42 = r_42[:, 1]

lassoeffect = hdm.rlassoEffects(x_42, y_42, index = [1, 2, 3, 50]);

hdm.r_print(lassoeffect)
hdm.r_summary(lassoeffect)
hdm.r_confint(lassoeffect)


using Highlights.Tokens, Highlights.Themes
abstract type ct <: AbstractTheme end


@theme ct Dict(
    # :name => "Tango",
    # :description => "A theme inspired by the Tango Icon Theme Guidelines.",
    # :comments => "Based on Tango theme from Pygments.",
    :style => S"fg: 990000",
    :tokens => Dict(
        COMMENT                => S"italic; fg: 8f5902",
        COMMENT_MULTILINE      => S"italic; fg: 8f5902",
        COMMENT_PREPROC        => S"italic; fg: 8f5902",
        COMMENT_SINGLE         => S"italic; fg: 8f5902",
        COMMENT_SPECIAL        => S"italic; fg: 8f5902",

        ERROR                  => S"fg: a40000; bg: ef2929",

        GENERIC                => S"fg: 000000",
        GENERIC_DELETED        => S"fg: a40000",
        GENERIC_EMPH           => S"italic; fg: 000000",
        GENERIC_ERROR          => S"fg: ef2929",
        GENERIC_HEADING        => S"bold; fg: 000080",
        GENERIC_INSERTED       => S"fg: 00a000",
        GENERIC_OUTPUT         => S"italic; fg: 16537e",
        GENERIC_PROMPT         => S"fg: 8f5902",
        GENERIC_STRONG         => S"bold; fg: 000000",
        GENERIC_SUBHEADING     => S"bold; fg: 800080",
        GENERIC_TRACEBACK      => S"bold; fg: a40000",

        KEYWORD                => S"bold; fg: 204a87",
        KEYWORD_CONSTANT       => S"bold; fg: 204a87",
        KEYWORD_DECLARATION    => S"bold; fg: 204a87",
        KEYWORD_NAMESPACE      => S"bold; fg: 204a87",
        KEYWORD_PSEUDO         => S"bold; fg: 204a87",
        KEYWORD_RESERVED       => S"bold; fg: 204a87",
        KEYWORD_TYPE           => S"bold; fg: 204a87",

        LITERAL                => S"fg: 000000",
        LITERAL_DATE           => S"fg: 000000",

        NAME                   => S"fg: 000000",
        NAME_ATTRIBUTE         => S"fg: c4a000",
        NAME_BUILTIN           => S"fg: 204a87",
        NAME_BUILTIN_PSEUDO    => S"fg: 3465a4",
        NAME_CLASS             => S"fg: 000000",
        NAME_CONSTANT          => S"fg: 000000",
        NAME_DECORATOR         => S"bold; fg: 5c35cc",
        NAME_ENTITY            => S"fg: ce5c00",
        NAME_EXCEPTION         => S"bold; fg: cc0000",
        NAME_FUNCTION          => S"bold; fg: 16537e",
        NAME_LABEL             => S"fg: f57900",
        NAME_NAMESPACE         => S"fg: 000000",
        NAME_OTHER             => S"fg: 000000",
        NAME_PROPERTY          => S"fg: 000000",
        NAME_TAG               => S"bold; fg: 204a87",
        NAME_VARIABLE          => S"fg: 000000",
        NAME_VARIABLE_CLASS    => S"fg: 000000",
        NAME_VARIABLE_GLOBAL   => S"fg: 000000",
        NAME_VARIABLE_INSTANCE => S"fg: 000000",

        NUMBER                 => S"bold; fg: 0000cf",
        NUMBER_FLOAT           => S"bold; fg: 0000cf",
        NUMBER_HEX             => S"bold; fg: 0000cf",
        NUMBER_INTEGER         => S"bold; fg: 0000cf",
        NUMBER_INTEGER_LONG    => S"bold; fg: 0000cf",
        NUMBER_OCT             => S"bold; fg: 0000cf",

        OPERATOR               => S"bold; fg: ce5c00",
        OPERATOR_WORD          => S"bold; fg: 204a87",

        OTHER                  => S"fg: 000000",

        PUNCTUATION            => S"bold; fg: 16537e",

        STRING                 => S"fg: 4e9a06",
        STRING_BACKTICK        => S"fg: 4e9a06",
        STRING_CHAR            => S"fg: 4e9a06",
        STRING_DOC             => S"italic; fg: 8f5902",
        STRING_DOUBLE          => S"fg: 4e9a06",
        STRING_ESCAPE          => S"fg: 4e9a06",
        STRING_HEREDOC         => S"fg: 4e9a06",
        STRING_INTERPOL        => S"fg: 4e9a06",
        STRING_OTHER           => S"fg: 4e9a06",
        STRING_REGEX           => S"fg: 4e9a06",
        STRING_SINGLE          => S"fg: 4e9a06",
        STRING_SYMBOL          => S"fg: 4e9a06",

        TEXT                   => S"",

        WHITESPACE             => S"underline; fg: f8f8f8",
    )
)

# bcbcbc

# weave("hdm.jmd"; doctype = "md2pdf", highlight_theme = ct)
# using Weave, Highlights

# @time weave("hdm.jmd"; doctype = "md2pdf", highlight_theme = Highlights.Themes.MonokaiMiniTheme, pandoc_options = ["--toc", "--toc-depth= 3", "--number-sections", "--self-contained"])
# @time weave("hdm.jmd"; doctype="pandoc2pdf", pandoc_options=["--toc", "--toc-depth= 1", "--number-sections", "--self-contained"], highlight_theme = Highlights.Themes.TangoTheme)
@time weave("hdm.jmd"; doctype="pandoc2pdf", pandoc_options=["--toc", "--toc-depth= 1", "--number-sections", "--self-contained"], highlight_theme = ct)
using RData, DataFrames, CodecXz

url = "https://github.com/cran/hdm/raw/master/data/pension.rda";
pension = load(download(url))["pension"];

y = pension[:, "tw"];
d = pension[:, "p401"];
z = pension[:, "e401"];
X = Matrix(pension[:, ["i2", "i3", "i4", "i5", "i6", "i7", "a2", "a3", "a4", "a5", "fsize", "hs", "smcol", "col", "marr", "twoearn", "db", "pira", "hown"]]);



# summary(pension.ate)
## Estimation and significance testing of the treatment effect
## Type: ATE
## Bootstrap: not applicable
## coeff. se. t-value p-value
## TE 10490 1920 5.464 4.67e-08 ***
## ---
## Signif. codes:
## 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


include("../src/HDMjl.jl")
x
ss = HDMjl.rlassoATET(X, d, y);
HDMjl.r_summary(ss);
# print.rlassoTE <- function(x, digits = max(3L, getOption("digits") - 3L), 
#                            ...) {
#   cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
#       "\n\n", sep = "")
#   if (length(x$te)) {
#     cat("Treatment Effect\n")
#     cat(paste("Type:", x$type), "\n")
#     cat("Value:\n")
#     print.default(format(x$te, digits = digits), print.gap = 2L, quote = FALSE)
#   } else cat("No treatment effect\n")
#   cat("\n")
#   invisible(x$te)
# }

# #' @rdname methods.rlassoTE
# #' @export

# summary.rlassoTE <- function(object, digits = max(3L, getOption("digits") - 
#                                                     3L), ...) {
#   if (length(object$te)) {
#     table <- matrix(NA, ncol = 4, nrow = 1)
#     rownames(table) <- "TE"
#     colnames(table) <- c("coeff.", "se.", "t-value", "p-value")
#     table[, 1] <- object$te
#     if (is.null(object$type_boot)) {
#       table[, 2] <- object$se
#     } else {
#       table[, 2] <- object$boot.se
#     }
#     table[, 3] <- table[, 1]/table[, 2]
#     table[, 4] <- 2 * pnorm(-abs(table[, 3]))
#     cat("Estimation and significance testing of the treatment effect", 
#         "\n")
#     cat(paste("Type:", object$type), "\n")
#     cat(paste("Bootstrap:", ifelse(is.null(object$type_boot), "not applicable", 
#                                    object$type_boot)), "\n")
#     printCoefmat(table, digits = digits, P.values = TRUE, has.Pvalue = TRUE)
#     cat("\n")
#   } else {
#     cat("No coefficients\n")
#   }
#   cat("\n")
#   invisible(table)
# }

# #' @rdname methods.rlassoTE
# #' @export

# confint.rlassoTE <- function(object, parm, level = 0.95, ...) {
#   n <- object$samplesize
#   k <- 1
#   cf <- object$te
#   pnames <- "TE"
#   a <- (1 - level)/2
#   a <- c(a, 1 - a)
#   fac <- qt(a, n - k)
#   pct <- format.perc(a, 3)
#   ci <- array(NA, dim = c(length(pnames), 2L), dimnames = list(pnames, 
#                                                                pct))
#   if (is.null(object$type_boot)) {
#     ses <- object$se
#   } else {
#     ses <- object$boot.se
#   }
#   ses <- object$se[parm]
#   ci[] <- cf[parm] + ses %o% fac
#   print(ci)
#   invisible(ci)
# }

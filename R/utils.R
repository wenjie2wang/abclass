## convert all the categories to {1, ..., k}
cat2z <- function(y) {
    fac_y <- as.factor(y)
    list(y = as.integer(fac_y),
         label = levels(fac_y),
         class_y = class(y))
}
## reverse convert
z2cat <- function(y, cat_y) {
    out <- cat_y$label[y]
    switch(cat_y$class_y,
           "integer" = as.integer(out),
           "numeric" = as.numeric(out),
           "factor" = as.factor(out),
           "character" = as.character(out))
}
## convert null to numeric(0)
null2num0 <- function(x) {
    if (is.null(x)) {
        return(numeric(0))
    }
    x
}
## convert null to numeric(0)
null2mat0 <- function(x) {
    if (is.null(x)) {
        return(matrix(numeric(0)))
    }
    x
}

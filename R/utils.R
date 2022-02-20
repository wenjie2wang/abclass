## convert all the categories to {0, ..., k - 1}
cat2z <- function(y) {
    fac_y <- as.factor(y)
    list(y = as.integer(fac_y) - 1L,
         label = levels(fac_y),
         class_y = class(y))
}

## reverse convert
z2cat <- function(y, cat_y) {
    out <- cat_y$label[y + 1L]
    switch(cat_y$class_y,
           "integer" = as.integer(out),
           "numeric" = as.numeric(out),
           "factor" = factor(out, levels = cat_y$label),
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

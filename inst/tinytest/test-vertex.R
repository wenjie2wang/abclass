vmat2 <- vertex(k = 2)
vmat3 <- vertex(k = 3)
vmat4 <- vertex(k = 4)
vmat5 <- vertex(k = 5)

## dimensions: (k-1) x k
expect_equal(dim(vmat2), c(1, 2))
expect_equal(dim(vmat3), c(2, 3))
expect_equal(dim(vmat4), c(3, 4))
expect_equal(dim(vmat5), c(4, 5))

## vertices lie on a unit sphere: column norms should be equal
col_norms <- function(m) apply(m, 2, function(v) sqrt(sum(v^2)))
expect_true(all(abs(diff(col_norms(vmat3))) < 1e-10))
expect_true(all(abs(diff(col_norms(vmat4))) < 1e-10))
expect_true(all(abs(diff(col_norms(vmat5))) < 1e-10))

## columns should sum to zero (centred simplex)
expect_true(all(abs(rowSums(vmat3)) < 1e-10))
expect_true(all(abs(rowSums(vmat4)) < 1e-10))

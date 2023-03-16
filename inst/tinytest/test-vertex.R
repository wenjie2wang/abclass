vmat2 <- vertex(k = 2)
vmat3 <- vertex(k = 3)

expect_equal(dim(vmat2), c(1, 2))
expect_equal(dim(vmat3), c(2, 3))

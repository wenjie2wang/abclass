if (requireNamespace("tinytest", quietly = TRUE) &&
    utils::packageVersion("tinytest") >= "1.2.2") {
    set.seed(808)
    tinytest::test_package("abclass", ncpu = NULL,
                           side_effects = TRUE)
}

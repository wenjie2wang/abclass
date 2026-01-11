if (requireNamespace("tinytest", quietly = TRUE) &&
    utils::packageVersion("tinytest") >= "1.2.2") {
    set.seed(808)
    is_at_home <- function() {
        identical(tolower(Sys.getenv("TT_AT_HOME")), "true")
    }
    if (is_at_home()) {
        ## only run the following tests if "at home" (not at cran)
        tinytest::test_package("abclass", ncpu = NULL,
                               testdir = "tests_with_suggests")
    }
    ## always run the following checks with no suggests
    tinytest::test_package("abclass", ncpu = NULL,
                           side_effects = TRUE)
}

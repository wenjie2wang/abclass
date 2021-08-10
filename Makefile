objects := $(wildcard R/*.R) DESCRIPTION
version := $(shell egrep "^Version:" DESCRIPTION | awk '{print $$NF}')
pkg := $(shell egrep "^Package:" DESCRIPTION | awk '{print $$NF}')
tar := $(pkg)_$(version).tar.gz
checkLog := $(pkg).Rcheck/00check.log

.PHONY: check
check: $(checkLog)

.PHONY: build
build: $(tar)

.PHONY: install
install: $(tar)
	R CMD INSTALL $(tar)

.PHONY: pkgdown
pkgdown:
	Rscript -e "library(methods); pkgdown::build_site();"

$(tar): $(objects)
	@$(RM) -rf src/RcppExports.cpp R/RcppExports.R
	@Rscript -e "library(methods);" \
	-e "Rcpp::compileAttributes()" \
	-e "devtools::document();";
	R CMD build .

$(checkLog): $(tar)
	R CMD check $(tar)

## make tags
.PHONY: tags
tags:
	Rscript -e "utils::rtags(path = 'R', ofile = 'TAGS')"
	gtags

.PHONY: clean
clean:
	@$(RM) -r *~ */*~ *.Rhistroy *.tar.gz src/*.so src/*.o \
	*.Rcheck/ *.Rout .\#* *_cache

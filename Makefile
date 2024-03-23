objects := $(wildcard R/*.R) DESCRIPTION
version := $(shell grep -E "^Version:" DESCRIPTION | awk '{print $$NF}')
pkg := $(shell grep -E "^Package:" DESCRIPTION | awk '{print $$NF}')
tar := $(pkg)_$(version).tar.gz
checkLog := $(pkg).Rcheck/00check.log

.PHONY: check
check: $(checkLog)

.PHONY: build
build: $(tar)

.PHONY: install
install:
	R CMD build .
	R CMD INSTALL $(tar)

.PHONY: pkgdown
pkgdown:
	Rscript -e "library(methods); pkgdown::build_site();"

.PHONY: deploy-pkgdown
deploy-pkgdown:
	@bash misc/deploy_docs.sh

.PHONY: readme
readme: README.md
README.md: README.Rmd
	@Rscript -e "rmarkdown::render('$<')"

.PHONY: update-timestamp
update-timestamp:
	@bash misc/update_timestamp.sh

## make tags
.PHONY: tags
tags:
	Rscript -e "utils::rtags(path = 'R', ofile = 'TAGS')"

$(tar): $(objects)
	@$(RM) -rf src/RcppExports.cpp R/RcppExports.R
	@Rscript -e "library(methods);" \
	-e "Rcpp::compileAttributes()" \
	-e "devtools::document();";
	@$(MAKE) update-timestamp
	R CMD build .

$(checkLog): $(tar)
	R CMD check --as-cran $(tar)

.PHONY: clean
clean:
	@$(RM) -r *~ */*~ *.Rhistroy *.tar.gz src/*.so src/*.o src/*.o.tmp \
	*.Rcheck/ *.Rout .\#* *_cache

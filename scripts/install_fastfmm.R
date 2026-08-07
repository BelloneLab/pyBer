#!/usr/bin/env Rscript

# Install the R package needed by pyBer's trial-level FLMM panel.
# Brutal truth: fastFMM is not available as a conda-forge package on Windows,
# so this must be installed from CRAN into the conda environment's R library.

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  install.packages.compile.from.source = "never"
)

install_if_missing <- function(package_name) {
  # Installs only the runtime dependency set. Suggested packages are avoided
  # because they make the lab install slower and less reproducible.
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(
      package_name,
      type = "binary",
      dependencies = c("Depends", "Imports", "LinkingTo")
    )
  }
}

# fastFMM loads these at runtime. Installing them explicitly avoids Windows CRAN
# binary edge cases where fastFMM can be present but fail during library().
install_if_missing("RcppParallel")
install_if_missing("fds")
install_if_missing("fastFMM")

suppressPackageStartupMessages(library(fastFMM))

cat("fastFMM", as.character(utils::packageVersion("fastFMM")), "is available\n")

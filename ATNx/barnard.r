# install.packages("Exact", repos="https://cloud.r-project.org")
library("Exact")

f_obs <- matrix(c(15, 0, 32, 47), nrow=2, ncol=2, byrow=TRUE)
result <- exact.test(f_obs, alternative="two.sided", method="csm", model="Binomial", cond.row=FALSE)
result

f_obs <- matrix(c(16, 13, 31, 34), nrow=2, ncol=2, byrow=TRUE)
result <- exact.test(f_obs, alternative="two.sided", method="csm", model="Binomial", cond.row=FALSE)
result
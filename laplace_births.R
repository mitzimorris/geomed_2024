library(posterior)
library(cmdstanr)
options(digits=3)
births_model = cmdstan_model("laplace.stan")
births_fit = births_model$sample(refresh=0)
print(as.data.frame(births_fit$summary(
  variables=c("theta", "theta_gt_half"), 
  c("mean", "median"), 
  quantiles = ~ quantile2(., probs = c(0.005, 0.995)))))



data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] x; // design matrix
}
transformed data {
  vector[N] log_E = log(E);
}
parameters {
  real beta0; // intercept
  vector[K] betas; // covariates
}
model {
  y ~ poisson_log(log_E + beta0 + x * betas);  // likelihood
  beta0 ~ std_normal();
  betas ~ std_normal();  // priors
}
// generated quantities {
//   vector[N] eta = log_E + beta0 + x * betas;
//   array[N] int y_rep;
//   if (max(eta) > 20) {
//     // avoid overflow in poisson_log_rng
//     print("max eta too big: ", max(eta));
//     for (n in 1:N)
//       y_rep[n] = -1;
//   } else {
//       for (n in 1:N)
//         y_rep[n] = poisson_log_rng(eta[n]);
//   }
//   //  vector[N] mu = exp(eta);  // estimated per-area rate
// }

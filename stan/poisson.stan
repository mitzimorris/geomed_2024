data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix
}
transformed data {
  vector[N] log_E = log(E);
}
parameters {
  real beta0; // intercept
  vector[K] betas; // covariates
}
model {
  y ~ poisson_log(log_E + beta0 + xs * betas);
  beta0 ~ std_normal();
  betas ~ std_normal();
}
generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  {
    vector[N] eta = log_E + beta0 + xs * betas;
    y_rep = max(eta) < 26 ? poisson_log_rng(eta) : rep_array(-1, N);
    for (n in 1:N) {
      log_lik[n] = poisson_log_lpmf(y[n] | eta[n]);
    }
  }
}

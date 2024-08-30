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
  beta0 ~ std_normal();  // priors
  betas ~ std_normal();
}
generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  {  // local block variables not recorded
    vector[N] eta = log_E + beta0 + x * betas;
    if (max(eta) > 26) {
      // avoid overflow in poisson_log_rng
      print("max eta too big: ", max(eta));
      for (n in 1:N) {
        y_rep[n] = -1;
        log_lik[n] = -1;
      }
    } else {
      for (n in 1:N) {
        y_rep[n] = poisson_log_rng(eta[n]);
        log_lik[n] = poisson_log_lpmf(y[n] | eta[n]);
      }
    }
  }
}

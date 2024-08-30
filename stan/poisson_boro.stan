data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  int<lower=1> N_boros;
  array[N] int<lower=1, upper=N_boros> boro_code;
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] x; // design matrix
}
transformed data {
  vector[N] log_E = log(E);
}
parameters {
  real beta0;  // global intercept
  vector[N_boros-1] boro_raw; // per-boro covariate
  vector[K] betas; // covariates
}
transformed parameters {
  vector[N_boros] boro = append_row(boro_raw, -sum(boro_raw));
}
model {
  y ~ poisson_log(log_E + beta0 + boro[boro_code] + x * betas);  // likelihood
  beta0 ~ std_normal();
  boro ~ std_normal();
  betas ~ std_normal();
}
generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  {
    vector[N] eta = log_E + beta0 + boro[boro_code] + x * betas;
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

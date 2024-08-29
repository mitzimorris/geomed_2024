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
  vector[N] theta; // heterogeneous random effects
  real<lower=0> sigma; // non-centered re variance 
}
model {
  y ~ poisson_log(log_E + beta0 + x * betas + theta * sigma);  // likelihood
  beta0 ~ std_normal();
  betas ~ std_normal();
  theta ~ std_normal();
  sigma ~ normal(0, 5);
}
generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  {
    vector[N] eta = log_E + beta0 + x * betas + theta * sigma;
    if (max(eta) > 20) {
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

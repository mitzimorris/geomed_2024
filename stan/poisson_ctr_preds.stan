data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] x; // design matrix
}
transformed data {
  vector[N] log_E = log(E);
  // center continuous predictors 
  vector[K] means_x;  // column means of x before centering
  matrix[N, K] x_centered;  // centered version of x
  for (k in 1:K) {
    means_x[k] = mean(x[, k]);
    x_centered[, k] = x[, k] - means_x[k];
  }
  print("means x", means_x);
  print("x_centered", x_centered[1:10, ]);
}
parameters {
  real beta0; // intercept
  vector[K] betas; // covariates
}
model {
  y ~ poisson_log(log_E + beta0 + x_centered * betas);   // centered data
  beta0 ~ std_normal();
  betas ~ std_normal();
}
generated quantities {
  real beta_intercept = beta0 - dot_product(means_x, betas);  // adjust intercept
  array[N] int y_rep;
  vector[N] log_lik;
  {
    vector[N] eta = log_E + beta0 + x_centered * betas;   // centered data
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
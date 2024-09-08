functions {
  /**
   * Compute ICAR, use soft-sum-to-zero constraint for identifiability
   *
   * @param phi vector of varying effects
   * @param adjacency parallel arrays of indexes of adjacent elements of phi
   * @param epsilon allowed variance for soft centering
   * @return ICAR log probability density
   * @reject if the the adjacency matrix does not have two rows
   */
  real standard_icar_lpdf(vector phi, array[ , ] int adjacency, real epsilon) {
    if (size(adjacency) != 2)
      reject("require 2rows for adjacency array;",
             " found rows = ", size(adjacency));
    return -0.5 * dot_self(phi[adjacency[1]] - phi[adjacency[2]])
      + normal_lupdf(sum(phi) | 0, epsilon * rows(phi));
  }
}
data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix

  // spatial structure
  int<lower = 0> N_edges;  // number of neighbor pairs
  array[2, N_edges] int<lower = 1, upper = N> neighbors;  // columnwise adjacent

  real tau; // scaling factor
}
transformed data {
  vector[N] log_E = log(E);
  // center continuous predictors 
  vector[K] means_xs;  // column means of xs before centering
  matrix[N, K] xs_centered;  // centered version of xs
  for (k in 1:K) {
    means_xs[k] = mean(xs[, k]);
    xs_centered[, k] = xs[, k] - means_xs[k];
  }
}
parameters {
  real beta0; // intercept
  vector[K] betas; // covariates
  real<lower=0, upper=1> rho; // proportion of spatial variance
  vector[N] phi; // spatial random effects
  vector[N] theta; // heterogeneous random effects
  real<lower = 0> sigma;  // scale of combined effects
}
transformed parameters {
  vector[N] gamma = (sqrt(1 - rho) * theta + sqrt(rho / tau) * phi);  // BYM2
}
model {
  y ~ poisson_log(log_E + beta0 + xs_centered * betas + gamma * sigma);
  rho ~ beta(0.5, 0.5);
  phi ~ standard_icar(neighbors, 0.001);
  beta0 ~ std_normal();
  betas ~ std_normal();
  theta ~ std_normal();
  sigma ~ std_normal();
}
generated quantities {
  real beta_intercept = beta0 - dot_product(means_xs, betas);  // adjust intercept
  array[N] int y_rep;
  vector[N] log_lik;
  {
    vector[N] eta = log_E + beta0 + xs_centered * betas + gamma * sigma;
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

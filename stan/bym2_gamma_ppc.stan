data {
  int<lower=0> N;
  // spatial structure
  int<lower = 0> N_edges;  // number of neighbor pairs
  array[2, N_edges] int<lower = 1, upper = N> neighbors;  // columnwise adjacent

  real tau; // scaling factor
}
parameters {
  real<lower=0, upper=1> rho; // proportion of spatial variance
  sum_to_zero_vector[N] phi;  // spatial effects
  vector[N] theta; // heterogeneous random effects
  real<lower = 0> sigma;  // scale of combined effects
}
transformed parameters {
  vector[N] gamma = sqrt(1 - rho) * theta + sqrt(rho * inv(tau)) * phi;  // BYM2
}
model {
  rho ~ beta(0.5, 0.5);
  target += -0.5 * dot_self(phi[neighbors[1]] - phi[neighbors[2]]); // ICAR prior
  theta ~ std_normal();
  sigma ~ std_normal();
}

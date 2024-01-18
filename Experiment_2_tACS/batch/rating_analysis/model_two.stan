data {
  int<lower=1> S; // number of subjects
  int<lower=1> J; // number of groups
  int<lower=1> N; // number of trials
  int<lower=1> K; // number of regressors
  int<lower=1,upper=J> grp[S]; // group info
  int<lower=1,upper=S> sub[N]; // subject info
  int<lower=0,upper=1> y[N];
  row_vector[K] X[N];
}

parameters {
  vector[K] beta_group[J];
  vector<lower=0>[K] sigma_s;
  vector[K] beta_subject[S];
}

model {
  vector[N] x_beta;
  sigma_s ~ cauchy(0,5);
  for (j in 1:J) beta_group[j] ~ normal(0, 100);
  for (s in 1:S) beta_subject[s] ~ normal(beta_group[grp[s]],sigma_s);
  for (n in 1:N) x_beta[n] = X[n] * beta_subject[sub[n]];
  y ~ bernoulli_logit(x_beta);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) log_lik[n] = bernoulli_logit_lpmf(y[n]| X[n] * beta_subject[sub[n]]);
}


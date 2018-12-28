import numpy as np
from scipy import stats

class Gaussian:
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma

	def pdf(self, datum):
		u = (datum - self.mu) / abs(self.sigma)
		y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y


class GaussianMixture:
	def __init__(self, data, mu_min=min(data), mu_max=max(data), sigma_min=.1, sigma_max=1, mix=.5):
		self.data = data
		self.one = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
		self.two = Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max))
		self.mix = mix

    def Estep(self):
		self.loglike = 0. # = log(p = 1)
		for datum in self.data:
			wp1 = self.one.pdf(datum) * self.mix
			wp2 = self.two.pdf(datum) * (1. - self.mix)
			den = wp1 + wp2
			wp1 /= den
			wp2 /= den
			self.loglike += log(wp1 + wp2)
			yield (wp1, wp2)

	def Mstep(self, weights):
		(left, rigt) = zip(*weights)
		one_den = sum(left)
		two_den = sum(rigt)
		self.one.mu = sum(w * d / one_den for (w, d) in zip(left, data))
		self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, data))
		self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)for (w, d) in zip(left, data)) / one_den)
		self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)for (w, d) in zip(rigt, data)) / two_den)
		self.mix = one_den / len(data)

	def pdf(self, x):
		return (self.mix)*self.one.pdf(x) + (1-self.mix)*self.two.pdf(x)


n_iterations = 5
best_mix = None
best_loglike = float('-inf')
mix = GaussianMixture(data)
for _ in range(n_iterations):
	try:
		mix.iterate(verbose=True)
		if mix.loglike > best_loglike:
			best_loglike = mix.loglike
			best_mix = mix
	except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
		pass



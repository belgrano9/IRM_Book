from scipy.stats import norm

class ExtendedVasicekModel(VasicekModel):
    def __init__(self, r0, k, theta, sigma):
        super().__init__(r0, k, theta, sigma)

    def simulate_forward_measure_short_rate(self, dt, T, s, t):
        N = int((t - s) / dt)
        r = np.zeros(N)
        r[0] = self.r0
        dW = np.sqrt(dt) * np.random.normal(size=N)
        
        MT = lambda s, t: (self.theta - self.sigma ** 2 / self.k ** 2) * (1 - np.exp(-self.k * (t - s))) + \
                          self.sigma ** 2 / (2 * self.k ** 2) * (np.exp(-self.k * (T - t)) - np.exp(-self.k * (T + t - 2 * s)))

        for i in range(1, N):
            r[i] = r[i-1] + (self.k * self.theta - self.k * r[i-1] + MT(s, s + i * dt)) * dt + self.sigma * dW[i]
        
        return r

    def forward_measure_conditional_mean_variance(self, s, t):
        MT = lambda s, t: (self.theta - self.sigma ** 2 / self.k ** 2) * (1 - np.exp(-self.k * (t - s))) + \
                          self.sigma ** 2 / (2 * self.k ** 2) * (np.exp(-self.k * (T - t)) - np.exp(-self.k * (T + t - 2 * s)))
        mean = self.r0 * np.exp(-self.k * (t - s)) + MT(s, t)
        variance = self.sigma ** 2 / (2 * self.k) * (1 - np.exp(-2 * self.k * (t - s)))
        return mean, variance

    def european_bond_option_price(self, t, T, S, X, option_type='call'):
        omega = 1 if option_type == 'call' else -1
        B_TS = (1 / self.k) * (1 - np.exp(-self.k * (S - T)))
        sigma_p = self.sigma * np.sqrt((1 - np.exp(-2 * self.k * (T - t))) / (2 * self.k)) * B_TS
        h = (1 / sigma_p) * np.log(self.bond_price(t, S) / (self.bond_price(t, T) * X)) + sigma_p / 2
        return omega * (self.bond_price(t, S) * norm.cdf(omega * h) - X * self.bond_price(t, T) * norm.cdf(omega * (h - sigma_p)))

# Example usage
extended_vasicek = ExtendedVasicekModel(r0=0.05, k=0.1, theta=0.05, sigma=0.01)
forward_rate_sim = extended_vasicek.simulate_forward_measure_short_rate(dt=0.01, T=1, s=0, t=1)
forward_mean, forward_variance = extended_vasicek.forward_measure_conditional_mean_variance(s=0, t=1)
bond_option_price = extended_vasicek.european_bond_option_price(t=0, T=1, S=2, X=0.9)

print(f"Simulated forward measure short rate: {forward_rate_sim}")
print(f"Forward measure conditional mean: {forward_mean}, Forward measure conditional variance: {forward_variance}")
print(f"European bond option price: {bond_option_price}")

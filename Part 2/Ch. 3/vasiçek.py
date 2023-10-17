import numpy as np

class VasicekModel:
    def __init__(self, r0, k, theta, sigma):
        self.r0 = r0
        self.k = k
        self.theta = theta
        self.sigma = sigma

    def simulate_short_rate(self, dt, T):
        N = int(T / dt)
        r = np.zeros(N)
        r[0] = self.r0
        dW = np.sqrt(dt) * np.random.normal(size=N)
        
        for i in range(1, N):
            r[i] = r[i-1] + self.k * (self.theta - r[i-1]) * dt + self.sigma * dW[i]
        
        return r

    def bond_price(self, t, T):
        B = (1 / self.k) * (1 - np.exp(-self.k * (T - t)))
        A = np.exp((self.theta - (self.sigma ** 2) / (2 * self.k ** 2)) * (B - T + t) - (self.sigma ** 2) / (4 * self.k) * B ** 2)
        return A * np.exp(-B * self.r0)

    def conditional_mean_variance(self, s, t):
        mean = self.r0 * np.exp(-self.k * (t - s)) + self.theta * (1 - np.exp(-self.k * (t - s)))
        variance = (self.sigma ** 2) / (2 * self.k) * (1 - np.exp(-2 * self.k * (t - s)))
        return mean, variance

# Example usage
vasicek = VasicekModel(r0=0.05, k=0.1, theta=0.05, sigma=0.01)
short_rate_sim = vasicek.simulate_short_rate(dt=0.01, T=1)
bond_price = vasicek.bond_price(t=0, T=1)
mean, variance = vasicek.conditional_mean_variance(s=0, t=1)

print(f"Simulated short rate: {short_rate_sim}")
print(f"Bond price: {bond_price}")
print(f"Conditional mean: {mean}, Conditional variance: {variance}")

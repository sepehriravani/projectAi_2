import random
import matplotlib.pyplot as plt

# محاسبه مقدار Fitness به صورت دستی
def calculate_fitness_manual(portfolio, returns, covariance):
    portfolio_return = sum(portfolio[i] * returns[i] for i in range(len(returns)))
    portfolio_variance = 0
    for i in range(len(returns)):
        for j in range(len(returns)):
            portfolio_variance += portfolio[i] * portfolio[j] * covariance[i][j]
    portfolio_risk = portfolio_variance ** 0.5
    return portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

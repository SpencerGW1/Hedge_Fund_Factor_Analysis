import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

FIG_DIR = Path("figures")


def save_fig(name):
    """Save the current figure to figures/<name>.png (used for the README)."""
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


class FactorModel:
    """Selects factors, fits and evaluates a factor regression for the hedge fund."""

    def __init__(self, filepath, target='Hedge Fund'):
        self.data = pd.read_csv(filepath, parse_dates=['perf_date'])
        self.target = target
        self.model = None
        self.factors = None
    
    def plot_fund_timeseries(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.data['perf_date'], self.data[self.target])
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        plt.xlabel('Date')
        plt.ylabel('Monthly return')
        plt.title('Hedge Fund Monthly Returns Over Time')
        plt.tight_layout()
        save_fig('fund_returns')
        plt.show()

    def plot_return_distribution(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.data[self.target], kde=True, bins=30)
        plt.xlabel('Monthly return')
        plt.title('Distribution of Hedge Fund Monthly Returns')
        plt.tight_layout()
        plt.show()

    def plot_factor_boxplots(self):
        factor_cols = self.data.drop(columns=['perf_date', self.target])
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=factor_cols, orient='h')   # horizontal = labels readable
        plt.title('Distribution of Factor Returns (post-cleaning)')
        plt.xlabel('Monthly return')
        plt.tight_layout()
        plt.show()

    def plot_factor_scatters(self, factor_list):
        for factor in factor_list:
            plt.figure(figsize=(8, 5))
            plt.scatter(self.data[factor], self.data[self.target], alpha=0.6)
            plt.axhline(0, color='grey', linestyle='--', linewidth=0.6)
            plt.axvline(0, color='grey', linestyle='--', linewidth=0.6)
            plt.xlabel(f'{factor} return')
            plt.ylabel(f'{self.target} return')
            plt.title(f'{self.target} vs {factor}')
            plt.tight_layout()
            plt.show()
    
    def correlation_heatmap(self):
        factor_cols = self.data.drop(columns=['perf_date', self.target])
        corr_matrix = factor_cols.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', annot_kws={'size': 7}, cmap='coolwarm', center=0)
        plt.title('Factor Correlation Matrix')
        plt.tight_layout()
        save_fig('factor_correlations')
        plt.show()
        return corr_matrix
    
    def compute_vif(self):
        # Crowding excluded because the first 24 observations are missing, reducing the usable sample size for multicollinearity diagnostics.
        X = self.data.drop(columns=['perf_date', self.target, 'Factor - Crowding'])
        X = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data['factor'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                           for i in range(X.shape[1])]
        return vif_data.sort_values('VIF', ascending=False)
    
    def fund_correlations(self):
        data_with_fund = self.data.drop(columns=['perf_date'])
        fund_corr = data_with_fund.corr()[self.target].drop(self.target)
        return fund_corr.reindex(fund_corr.abs().sort_values(ascending=False).index)
    
    def lasso_selection(self):
        # Standardise predictors before LASSO so the penalty is applied consistently across factors measured on different scales.
        X = self.data.drop(columns=['perf_date', self.target, 'Factor - Crowding'])
        y = self.data[self.target]
        X_scaled = StandardScaler().fit_transform(X)
        lasso = LassoCV(cv=5, random_state=0).fit(X_scaled, y)
        coefs = pd.DataFrame({'factor': X.columns, 'coef': lasso.coef_})
        return coefs.reindex(coefs['coef'].abs().sort_values(ascending=False).index)
    
    def fit_ols(self, factor_list, cov_type='nonrobust', cov_kwds=None):
        X = self.data[factor_list]
        y = self.data[self.target]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        return model
    
    def compare_candidate_models(self):
        candidates = {
            'A': ['Factor - Value vs Growth', 'Factor - Credit',
                  'Factor - Momentum', 'Factor - Emerging Markets'],
            'B': ['Factor - Value vs Growth', 'Factor - Momentum', 'Factor - Credit'],
            'C': ['Factor - Value vs Growth', 'Factor - Credit'],
            'D': ['Factor - Value vs Growth', 'Factor - Credit',
                  'Factor - Emerging Markets'],
        }
        for name, factor_list in candidates.items():
            model = self.fit_ols(factor_list)
            print(f"\n=== Candidate {name}: {factor_list} ===")
            print(model.summary())
        
    def evaluate(self):
        if self.model is None:
            raise ValueError("No model fitted. Call fit_ols and assign to self.model first.")

        # residual plots
        # fit=True standardises residuals so they are comparable to the 45-degree normal reference line
        sm.qqplot(self.model.resid, line='45', fit=True)
        plt.title('Q-Q Plot of Residuals')
        save_fig('residuals_qq')
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(self.model.fittedvalues, self.model.resid, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
        save_fig('residuals_vs_fitted')
        plt.show()

        # formal heteroscedasticity test
        bp_test = het_breuschpagan(self.model.resid, self.model.model.exog)
        print('Breusch-Pagan p-value:', bp_test[1])
        

class StrategyAnalysis:
    """Compares the fund against a factor-replicating portfolio: Sharpe, risk metrics and beta stability."""

    def __init__(self, data, model, factors, target='Hedge Fund'):
        self.data = data
        self.model = model
        self.factors = factors
        self.target = target
        self.fund_returns = data[target]

    def factor_portfolio(self):
        # Remove alpha (intercept) so the portfolio represents only the systematic factor exposures captured by the regression.
        return self.model.fittedvalues - self.model.params['const']

    def sharpe_ratio(self, returns, rf=0, periods_per_year=12):
        excess = returns - rf
        return (excess.mean() / excess.std()) * (periods_per_year ** 0.5)

    def compare_sharpe(self):
        factor_port = self.factor_portfolio()
        results = {
            'Fund Sharpe': self.sharpe_ratio(self.fund_returns),
            'Factor Portfolio Sharpe': self.sharpe_ratio(factor_port),
            'Fund mean monthly return': self.fund_returns.mean(),
            'Factor Portfolio mean monthly return': factor_port.mean(),
        }
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        return results

    def plot_cumulative_returns(self):
        """Plot growth of 1 unit invested in the fund vs the factor-replicating portfolio."""
        factor_port = self.factor_portfolio()
        plt.figure(figsize=(12, 5))
        plt.plot(self.data['perf_date'], (1 + self.fund_returns).cumprod(), label='Hedge Fund')
        plt.plot(self.data['perf_date'], (1 + factor_port).cumprod(), label='Factor portfolio')
        plt.xlabel('Date')
        plt.ylabel('Growth of 1')
        plt.title('Cumulative Returns: Fund vs Factor-Replicating Portfolio')
        plt.legend()
        plt.tight_layout()
        save_fig('cumulative_returns')
        plt.show()

    def annualised_vol(self, returns, periods_per_year=12):
        return returns.std() * (periods_per_year ** 0.5)

    def max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return ((cumulative / running_max) - 1).min()

    def downside_deviation(self, returns, threshold=0, periods_per_year=12):
        downside = returns[returns < threshold]
        return downside.std() * (periods_per_year ** 0.5)

    def risk_metrics(self):
        factor_port = self.factor_portfolio()
        metrics = pd.DataFrame({
            'Fund': [
                self.annualised_vol(self.fund_returns),
                self.max_drawdown(self.fund_returns),
                self.downside_deviation(self.fund_returns),
                self.fund_returns.skew(),
                self.fund_returns.kurt()
            ],
            'Factor Portfolio': [
                self.annualised_vol(factor_port),
                self.max_drawdown(factor_port),
                self.downside_deviation(factor_port),
                factor_port.skew(),
                factor_port.kurt()
            ]
        }, index=['Annualised Vol', 'Max Drawdown', 'Downside Deviation',
                  'Skew', 'Excess Kurtosis'])
        return metrics

    def rolling_betas(self, window=36):
        X = sm.add_constant(self.data[self.factors])
        y = self.data[self.target]
        rolling_model = RollingOLS(y, X, window=window).fit()
        betas = rolling_model.params

        plt.figure(figsize=(12, 6))
        for factor in self.factors:
            plt.plot(self.data['perf_date'], betas[factor], label=f'{factor} beta')
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        plt.xlabel('Date')
        plt.ylabel(f'Rolling beta ({window}-month window)')
        plt.title('Rolling Factor Betas Over Time')
        plt.legend()
        plt.tight_layout()
        save_fig('rolling_betas')
        plt.show()
        return betas

    def test_stationarity(self, window=36):
        betas = self.rolling_betas(window=window)
        for factor in self.factors:
            series = betas[factor].dropna()
            # Test whether rolling factor exposures are stationary and revert to a stable long-run mean over time.
            result = adfuller(series)
            print(f"{factor}: ADF stat = {result[0]:.3f}, p-value = {result[1]:.3f}")

if __name__ == "__main__":
    fm = FactorModel("cleaned_data.csv")
    # EDA
    fm.plot_fund_timeseries()
    fm.plot_return_distribution()
    fm.plot_factor_boxplots()
    fm.plot_factor_scatters(['Factor - Value vs Growth', 'Factor - Credit'])
    # multicollinearity diagnostics
    fm.correlation_heatmap()
    print(fm.compute_vif())
    print(fm.fund_correlations())
    print(fm.lasso_selection())
    fm.compare_candidate_models()

    # fit and evaluate the chosen two-factor model (HAC robust)
    # HAC standard errors are used to account for potential autocorrelation and heteroscedasticity in monthly returns.
    fm.factors = ['Factor - Value vs Growth', 'Factor - Credit']
    fm.model = fm.fit_ols(fm.factors, cov_type='HAC', cov_kwds={'maxlags': 1})
    print(fm.model.summary())
    fm.evaluate()

    # fund vs factor portfolio: Sharpe, risk metrics and beta stationarity
    sa = StrategyAnalysis(fm.data, fm.model, fm.factors)
    sa.compare_sharpe()
    sa.plot_cumulative_returns()
    print(sa.risk_metrics())
    sa.test_stationarity()
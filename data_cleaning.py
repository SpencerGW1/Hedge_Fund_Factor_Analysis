import pandas as pd
import numpy as np


class DataCleaner:
    """Loads and cleans the hedge fund and factor returns data.

    Handles the four data quality issues identified in the returns sheet:
    out-of-order date blocks, scaling/units errors in two factor columns,
    leading placeholder zeros in the Crowding factor, and zero-return months
    in the fund series.
    """

    def __init__(self, filepath, sheet_name='returns data', threshold=5):
        # load raw data and store the numeric (factor + fund) columns for scanning
        self.data = pd.read_excel(filepath, sheet_name=sheet_name)
        self.threshold = threshold
        self.numeric_cols = [col for col in self.data.columns if col != 'perf_date']

    def inspect(self):
        """Report basic structure: shape, dtypes, and duplicate rows/dates."""
        print("Shape:", self.data.shape)
        print("\nData types:\n", self.data.dtypes)
        print("\nDuplicate rows:", self.data.duplicated().sum())
        print("Duplicate dates:", self.data['perf_date'].duplicated().sum())

    def check_dates(self):
        """Check date ordering and completeness.

        Inspection showed the 2018 and 2019 blocks were out of chronological
        order; all 195 monthly dates are present with no gaps or duplicates.
        """
        monotonic = self.data['perf_date'].is_monotonic_increasing
        print("Dates monotonic increasing:", monotonic)

        # locate any row whose date is not later than the previous row
        breaks = self.data['perf_date'].diff() <= pd.Timedelta(0)
        print("Break locations:\n", breaks[breaks])

        # compare actual dates against the expected monthly sequence
        expected = pd.date_range(start=self.data['perf_date'].min(),
                                 end=self.data['perf_date'].max(), freq='ME')
        missing = set(expected) - set(self.data['perf_date'])
        extra = set(self.data['perf_date']) - set(expected)
        print("Expected months:", len(expected), "Actual rows:", len(self.data))
        print("Missing dates:", missing)
        print("Unexpected dates:", extra)

    def fix_date_order(self):
        """Sort rows chronologically (whole rows move together, values intact)."""
        self.data = self.data.sort_values('perf_date').reset_index(drop=True)
        print("Dates sorted into chronological order.")

    def scan_outliers_zscore(self):
        """Flag values more than `threshold` standard deviations from the mean."""
        print("\n=== Z-Score Scan ===")
        for col in self.numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            z_scores = (self.data[col] - mean) / std
            flagged = z_scores[z_scores.abs() > self.threshold]
            if len(flagged) > 0:
                print(f"Column: {col}")
                print(flagged, "\n")

    def scan_outliers_iqr(self):
        """Flag values beyond 1.5*IQR of the quartiles.

        More robust than z-score when multiple extreme values exist in a column
        (extreme values inflate the std and can mask each other under z-score).
        """
        print("\n=== IQR Scan ===")
        for col in self.numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            flagged = self.data[col][(self.data[col] < lower) | (self.data[col] > upper)]
            if len(flagged) > 0:
                print(f"Column: {col}")
                print(flagged, "\n")

    def fix_scaling_errors(self, divisor=1_000_000, cutoff=1):
        """Rescale units/scaling errors.

        Genuine monthly returns never exceed +/-1, so any |value| > cutoff is a
        units error. Division by 1e6 was confirmed to bring all affected cells
        (Value vs Growth and Interest Rates) back into the normal range.
        """
        for col in self.numeric_cols:
            mask = self.data[col].abs() > cutoff
            self.data.loc[mask, col] = self.data.loc[mask, col] / divisor
        print("Scaling errors rescaled.")

    def handle_crowding(self, col='Factor - Crowding'):
        """Convert leading placeholder zeros to NaN.

        Crowding is zero for the first 24 months (factor not yet tracked, 2006-07)
        then populated. These leading zeros are missing data, not true zeros.
        """
        n_zeros_before = (self.data[col] == 0).sum()
        first_non_zero = (self.data[col] != 0).idxmax()
        self.data.loc[:first_non_zero - 1, col] = np.nan
        print(f"Crowding: {n_zeros_before} leading zeros -> NaN; "
              f"now {self.data[col].isna().sum()} NaNs.")

    def check_hedge_fund_zeros(self):
        """Report the two zero-return fund months for review.

        Both fall in low-volatility periods (Feb 2007, Jun 2019) where a flat
        monthly return is plausible, so they are retained rather than treated
        as missing.
        """
        zeros = self.data[self.data['Hedge Fund'] == 0]
        print("Hedge Fund zero-return months:\n", zeros[['perf_date', 'Hedge Fund']])
        print("Retained as plausible flat months in low-volatility periods.")

    def save(self, path="cleaned_data.csv"):
        """Write the cleaned dataset to CSV for the analysis stage."""
        self.data.to_csv(path, index=False)
        print(f"Cleaned data saved to {path}")

    def clean(self):
        """Run the full cleaning pipeline in order and return the cleaned data."""
        self.inspect()
        self.check_dates()
        self.fix_date_order()
        self.scan_outliers_zscore()
        self.scan_outliers_iqr()
        self.fix_scaling_errors()
        self.handle_crowding()
        self.check_hedge_fund_zeros()
        self.save()
        return self.data


if __name__ == "__main__":
    cleaner = DataCleaner('data.xlsx')   # place data.xlsx in the project folder
    cleaner.clean()
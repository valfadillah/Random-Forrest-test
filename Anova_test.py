import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load dataset
df = pd.read_csv("sleep_health_lifestyle_dataset.csv")

# Cleaning name in dataset
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
df.rename(columns={"quality_of_sleep_(scale:_1-10)": "quality_of_sleep"}, inplace=True)

# Anova test
anova_model = ols('quality_of_sleep ~ C(bmi_category)', data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

# Ekstrak p-value
anova_p_value = anova_table["PR(>F)"][0]

# Anova preview
print("\n=== Hasil Uji ANOVA ===")
print(anova_table)
print(f"\nP-Value: {anova_p_value:.5f}")

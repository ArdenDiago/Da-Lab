import pandas as pd
from scipy import stats

titanic_data = pd.read_csv("./data.csv")

text_formatter = (
    lambda test_name, test_pair: f"\n{test_name} T-Test\n\tT-statistic: {test_pair.statistic}\n\tP-Value: {test_pair.pvalue}"
)


# One-Sampel T-Test against age
hypothetical_mean_age = 30
ttest_one_sampel = stats.ttest_1samp(titanic_data['Age'].dropna(), hypothetical_mean_age)

print(text_formatter('One-Sample', ttest_one_sampel))

# Two Independent Sample
male_age = titanic_data[titanic_data['Sex'] == 'male']['Age'].dropna()
female_age = titanic_data[titanic_data['Sex'] == 'female']['Age'].dropna()

ttest_two_sample = stats.ttest_ind(male_age, female_age)
print(text_formatter("Two Independent Sample", ttest_two_sample))

# Paired T-Test
before_fare = titanic_data['Fare'].dropna()
after_fares= before_fare * 1.2
ttest_pair = stats.ttest_rel(before_fare, after_fares)
print(text_formatter('Paired T-Test', ttest_pair))
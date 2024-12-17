import pandas as pd
from scipy import stats

text_froma = lambda test_name, result: f"{test_name} T-Test\n\tT-statistic: {result.statistic}\n\tP-Value: {result.pvalue}"


td = pd.read_csv('./data.csv')

# one sample
Hypothetical_age = 30
ages = td['Age'].dropna()
one_sample = stats.ttest_1samp(ages, Hypothetical_age)
print(text_froma('One Sample', one_sample))

# two independet
male = td[td['Sex'] == 'Male']['Age'].dropna()
female = td[td['Sex'] == 'Feale']['Age'].dropna()

two_independent = stats.ttest_ind(male, female)
print(text_froma('two independent Sample', two_independent))

befor_fair = td['Fare'].dropna()
after = befor_fair * 1.5

pari = stats.ttest_rel(befor_fair, after)
print(text_froma('Paired', pari))
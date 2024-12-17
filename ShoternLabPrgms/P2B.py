import pandas as pd
from scipy import stats


def status_res(val1, val2, reason):
    return (
        "Reject the null hypothesis: {reason}.".format(reason=reason[0])
        if val1 > val2
        else "Fail to reject the null hypothesis: {reason}.".format(reason=reason[1])
    )


# Function for One-Way ANOVA
def perform_anova(groups, alpha=0.05):
    anova_result = stats.f_oneway(*groups)
    dfn = len(groups) - 1  # Degrees of freedom for numerator
    dfd = sum(len(group) for group in groups) - len(
        groups
    )  # Degrees of freedom for denominator
    f_critical = stats.f.ppf(1 - alpha, dfn, dfd)
    status = status_res(
        anova_result.statistic,
        f_critical,
        [
            "Significant differences exist among groups",
            "No significant differences among groups",
        ],
    )
    print("\nOne-Way ANOVA:")
    print("F-statistic:", anova_result.statistic)
    print("F-critical value:", f_critical)
    print(status)


# Function for Chi-Square Test
def perform_chi2_test(contingency_table, alpha=0.05):
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    chi2_critical = stats.chi2.ppf(1 - alpha, dof)
    status = status_res(
        chi2_stat,
        chi2_critical,
        ["Significant association exists", "No significant association"],
    )

    print("\nChi-Square Test:")
    print("Chi2 Statistic:", chi2_stat)
    print("Chi2 Critical value:", chi2_critical)
    print("Degrees of Freedom:", dof)
    print(status)


# Load Titanic dataset
titanic_data = pd.read_csv("./data.csv")

# Perform ANOVA: Impact of passenger class on age
grouped_ages = [
    titanic_data[titanic_data["Pclass"] == 1]["Age"].dropna(),
    titanic_data[titanic_data["Pclass"] == 2]["Age"].dropna(),
    titanic_data[titanic_data["Pclass"] == 3]["Age"].dropna(),
]
perform_anova(grouped_ages)

# Perform Chi-Square Test: Relationship between survival status and passenger class
contingency_table = pd.crosstab(titanic_data["Pclass"], titanic_data["Survived"])
perform_chi2_test(contingency_table)

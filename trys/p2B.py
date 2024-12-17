import pandas as pd
from scipy import stats



def status_res(val1, val2, reason):
    return (
        f"rejected the null hypothesis: {reason[0]}"
        if val1 > val2
        else f"Fail to reject the null hypothesis: {reason[1]}"
    )


def perform_anova(groups, alpha=0.05):
    anova_result = stats.f_oneway(*groups)
    dfn = len(groups) - 1
    dfd = sum(len(group) for group in groups) - len(groups)
    f_critical = stats.f.ppf(1 - alpha, dfn, dfd)

    status = status_res(
        anova_result.statistic,
        f_critical,
        ["Significant Difference exists amoung groups", "No sigifinant Difference"],
    )

    print(
        f"\nOne-Way Anova\n\tF-statistic: {anova_result.statistic}\n\tF-Critical Value: {f_critical}\n{status}"
    )


def chi2_test(contingency_table, alpha=0.05):
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    chi2_critical = stats.chi2.ppf(1 - alpha, dof)
    status = status_res(
        chi2_stat,
        chi2_critical,
        ["Signigicant association exists", "No significatnt association"],
    )
    print(
        f"\nchi-square Anova\n\tF-statistic: {chi2_stat}\n\tF-Critical Value: {chi2_critical}\n\tDegree of Freedom: {dof}\n{status}"
    )


td = pd.read_csv("./data.csv")

td = td.dropna(subset=["Age", "Pclass_1", "Survived"])

grouped_age = [
    [td[td["Pclass_1"] == 1]["Age"]],
    td[td["Pclass_2"] == 2]["Age"],
    td[td["Pclass_3"] == 3]["Age"],
]

perform_anova(grouped_age)

ct = pd.crossTab(td['Pclass_1'], td['Survived'])
chi2_test(ct)
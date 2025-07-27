import pandas as pd
import pickle
import statsmodels.api as sm
import streamlit as st

# Name the app
st.header('Predicting the health consequences of HRH expansion under a limited budget in 2025-2034, Malawi')

st.subheader("Inputs of the incremental budget allocation", divider='rainbow')

# Take inputs
p_clinical = st.number_input(label='Enter the budget proportion for Clinical cadre (in %)',
                             min_value=0.0, max_value=100.0, value=22.0, step=1.0)
p_dcsa = st.number_input(label='Enter the budget proportion for DCSA cadre (in %)',
                         min_value=0.0, max_value=100.0, value=23.0, step=1.0)
p_nursing = st.number_input(label='Enter the budget proportion for Nursing and Midwifery cadre (in %)',
                            min_value=0.0, max_value=100.0, value=45.0, step=1.0)
p_pharmacy = st.number_input(label='Enter the budget proportion for Pharmacy cadre (in %)',
                             min_value=0.0, max_value=100.0, value=3.0, step=1.0)
p_other = st.number_input(label='Enter the budget proportion for Other cadre (in %)',
                          min_value=0.0, max_value=100.0, value=7.0, step=1.0)
p = [p_clinical/100, p_dcsa/100, p_nursing/100, p_pharmacy/100, p_other/100]

# Check inputs if summing up to 1 and re-take the inputs
if abs(sum(p) - 1) > 0:
    st.error('The 5 input proportions do not sum up to 100%. Please re-input.')

# Choose the prediction setting
setting = st.radio('Select the setting for the prediction: ',
                   ('Main analysis',
                    'More budget',
                    'Less budget',
                    'Default consumable availability',
                    'Maximal health system function',
                    )
                   )


# Define functions to transform inputs to average increase rates
def increase_rate(R=0.042, input=p):
    # the current cost fractions for each cadre, in order of Clinical, DCSA, Nursing and Midwifery, Pharmacy, Other
    f = [0.2178, 0.2349, 0.4514, 0.0269, 0.0690]
    # calculate the increase rate x
    x = [0, 0, 0, 0, 0]
    for i in range(5):
        x[i] = (1 + input[i] * ((1+R) ** 10 - 1) / f[i]) ** (1/10) - 1
        x[i] = 100 * x[i]
    return x


# Define functions to calculate the health outcome
def predict_with_ci(input=p, budget_growth_rate=0.042, setting='main'):
    # transform the input strategy to increase rate
    rate = increase_rate(R=budget_growth_rate, input=input)
    predictor = pd.DataFrame(
        data=[rate],
        columns=['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy', 'Other']
    )
    predictor = sm.add_constant(predictor, has_constant='add')
    # get the regression model
    file_path = f'predict_health_outcome/reg_model_{setting}.pkl'
    with open(file_path, 'rb') as f:
        est = pickle.load(f)
    # get the prediction
    pred = est.get_prediction(predictor)
    pred_summary = pred.summary_frame(alpha=0.05)
    mean_dalys_in_million = pred_summary.loc[0, 'mean'].round(2)
    ci_lower = pred_summary.loc[0, 'mean_ci_lower'].round(2)
    ci_upper = pred_summary.loc[0, 'mean_ci_upper'].round(2)
    pi_lower = pred_summary.loc[0, 'obs_ci_lower'].round(2)
    pi_upper = pred_summary.loc[0, 'obs_ci_upper'].round(2)

    return mean_dalys_in_million, ci_lower, ci_upper, pi_lower, pi_upper


# calculate the outcomes
if setting == 'Main analysis':
    outcomes = predict_with_ci(input=p, budget_growth_rate=0.042, setting='main')
elif setting == 'More budget':
    outcomes = predict_with_ci(input=p, budget_growth_rate=0.058, setting='more_budget')
elif setting == 'Less budget':
    outcomes = predict_with_ci(input=p, budget_growth_rate=0.026, setting='less_budget')
elif setting == 'Default consumable availability':
    outcomes = predict_with_ci(input=p, budget_growth_rate=0.042, setting='default_cons')
elif setting == 'Maximal health system function':
    outcomes = predict_with_ci(input=p, budget_growth_rate=0.042, setting='max_hs_func')

# Check the predicting button and print outcomes
st.subheader("The output of health outcome", divider='green')
if st.button('Predict'):
    # print the health outcome
    st.success(f"The predicted DALYs is **{outcomes[0]}** million in the 10 year period of 2025-2034. "
               f"The 95% Confidence Interval is **[{outcomes[1]}, {outcomes[2]}]** and 95% Prediction Interval is "
               f"**[{outcomes[3]}, {outcomes[4]}]**.")

# Markdown
st.subheader("The explanation of the predictor", divider="orange")
st.markdown(
    """
This predictor outputs the estimated health outcome of any strategy that allocates a limited incremental budget to expand
multiple HCW cadres in the period between 2025 and 2034.

Each HRH expansion strategy is determined by inputs of five percentage numbers for each of Clinical, DCSA, 
Nursing and Midwifery, Pharmacy and Other cadres that sum up to 100%. Each number represents the proportion of the 
limited incremental budget that is allocated to each cadre for expansion each year. The health outcomes are measured 
in Disability Adjusted Life Years (DALYs). (The default proportions in this predictor represent the current HRH salary 
distribution across cadres.)

The prediction can be done in 5 settings:\\
    (1) main analysis - \\
        annual budget growth rate = 4.2%, perfect consumable availability, default health system function,\\
    (2) sensitivity analysis with more budget -\\
        annual budget growth rate = 5.8%, perfect consumable availability, default health system function,\\
    (3) sensitivity analysis with less budget -\\
        annual budget growth rate = 2.6%, perfect consumable availability, default health system function,\\
    (4) sensitivity analysis with default consumable availability - \\
        annual budget growth rate = 4.2%, default consumable availability, default health system function,\\
    (5) sensitivity analysis with maximal health system function - \\
        annual budget growth rate = 4.2%%, perfect consumable availability, maximal health system function.

By "perfect" or "default" consumable availability, we assume the consumables can always be provided upon request or 
the availability is informed by available data. By "maximal" or "default" health system function, we assume perfect 
diagnosis, referral practice and HCW competence or these parameters are informed by available data. In all settings,
we assume fixed salaries for HCW cadres over the period.

Please refer to the paper for full details.

    """
)

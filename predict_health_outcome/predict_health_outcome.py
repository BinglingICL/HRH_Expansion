import pandas as pd
import pickle
import statsmodels.api as sm
import streamlit as st

# Name the app
st.header('Predicting the health consequences of HRH expansion under a limited budget in 2025-2034, Malawi')

st.subheader("The input of the predictor", divider='rainbow')

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
def main_analysis_with_ci(input=p):
    # transform the input strategy to increase rate
    rate = increase_rate(R=0.042, input=input)
    predictor = pd.DataFrame(
        data=[rate],
        columns=['Clinical', 'DCSA', 'Nursing_and_Midwifery', 'Pharmacy', 'Other']
    )
    predictor = sm.add_constant(predictor, has_constant='add')
    # get the regression model
    with open('predict_health_outcome/reg_model_main.pkl', 'rb') as f:
        est = pickle.load(f)
    # get the prediction
    pred = est.get_prediction(predictor)
    pred_summary = pred.summary_frame(alpha=0.05)
    mean_dalys_in_million = pred_summary.loc[0, 'mean'].round(2)
    ci = [pred_summary.loc[0, 'mean_ci_lower'].round(2), pred_summary.loc[0, 'mean_ci_upper'].round(2)]
    pi = [pred_summary.loc[0, 'obs_ci_lower'].round(2), pred_summary.loc[0, 'obs_ci_upper'].round(2)]

    return mean_dalys_in_million, ci, pi


def main_analysis(input=p):
    # the linear predicting model
    const = 100.8007
    coefs = [-0.9464, -0.3928, - 0.9712, -0.2536, -0.1851]
    # get increase rates in percentage
    rate = increase_rate(R=0.042, input=input)
    # calculate the health outcome
    dalys_incurred_in_million = sum([coefs[i] * rate[i] for i in range(5)]) + const
    return dalys_incurred_in_million


def more_budget(input=p):
    # the linear predicting model
    const = 100.4043
    coefs = [-0.7492, -0.2993, -0.7124, -0.2190, -0.1472]
    # get increase rates in percentage
    rate = increase_rate(R=0.058, input=input)
    # calculate the health outcome
    dalys_incurred_in_million = sum([coefs[i] * rate[i] for i in range(5)]) + const
    return dalys_incurred_in_million


def less_budget(input=p):
    # the linear predicting model
    const = 102.3225
    coefs = [-1.3148, -0.6650, -1.5711, -0.3221, -0.2774]
    # get increase rates in percentage
    rate = increase_rate(R=0.026, input=input)
    # calculate the health outcome
    dalys_incurred_in_million = sum([coefs[i] * rate[i] for i in range(5)]) + const
    return dalys_incurred_in_million


def default_cons(input=p):
    # the linear predicting model
    const = 118.1100
    coefs = [-0.7652, -0.2997, -0.8217, -0.1759, -0.350]
    # get increase rates in percentage
    rate = increase_rate(R=0.042, input=input)
    # calculate the health outcome
    dalys_incurred_in_million = sum([coefs[i] * rate[i] for i in range(5)]) + const
    return dalys_incurred_in_million


def max_hs_func(input=p):
    # the linear predicting model
    const = 140.3744
    coefs = [-1.4983, -0.4446, -2.0231, -0.220, -0.2145]
    # get increase rates in percentage
    rate = increase_rate(R=0.042, input=input)
    # calculate the health outcome
    dalys_incurred_in_million = sum([coefs[i] * rate[i] for i in range(5)]) + const
    return dalys_incurred_in_million


# calculate the outcomes
if setting == 'Main analysis':
    outcomes = main_analysis_with_ci(input=p)
elif setting == 'Sensitivity analysis with more budget':
    outcomes = more_budget(input=p)
elif setting == 'Sensitivity analysis with less budget':
    outcomes = less_budget(input=p)
elif setting == 'Sensitivity analysis with default consumable availability':
    outcomes = default_cons(input=p)
elif setting == 'Sensitivity analysis with maximal health system function':
    outcomes = max_hs_func(input=p)

# Check the predicting button and print outcomes
st.subheader("The predicted outcome", divider='green')
if st.button('Predict the health outcome'):
    # print the health outcome
    st.success(f"The predicted DALYs is **{outcomes[0]}** million in the 10 year period of 2025-2034. "
               f"The 95% Confidence Interval is **{outcomes[1]}** and 95% Prediction Interval is **{outcomes[2]}**.")

# Markdown
st.subheader("The explanation of the predictor", divider="orange")
st.markdown(
    """
This predictor outputs the estimated health outcome of any strategy that allocates a limited extra budget to expand
multiple HCW cadres in the period between 2025 and 2034.

Each HRH expansion strategy is determined by inputs of five percentage numbers for each of Clinical, DCSA, 
Nursing and Midwifery, Pharmacy and Other cadres that sum up to 100%. Each number represents the proportion of the 
limited additional budget that is allocated to each cadre for expansion each year. The health outcomes are measured 
in Disability Adjusted Life Years (DALYs).

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
        annual budget growth rate = 4.2%%, perfect consumable availability, maximal health system function.\\

By "perfect" or "default" consumable availability, we assume the consumables can always be provided upon request or 
the availability is informed by available data. By "maximal" or "default" health system function, we assume perfect 
diagnosis, referral practice and HCW competence or these parameters are informed by available data. In all settings,
we assume fixed salaries for HCW cadres over the period.

Please refer to the paper for full details.

    """
)

import streamlit as st
import numpy as np
from scipy.stats import t

st.set_page_config(page_title="T-Test Calculator", layout="centered")

st.title("One Sample T-Test Calculator")

st.write("Enter sample data separated by commas.")

# Inputs
data_input = st.text_area("Sample values (e.g. 12, 15, 14, 10)")
mu = st.number_input("Hypothesized Mean (μ₀)", value=0.0)
alpha = st.number_input("Significance Level (α)", value=0.05)
alternative = st.selectbox("Alternative Hypothesis",
                           ["two-sided", "greater", "less"])

# Button
if st.button("Calculate"):

    if data_input.strip() == "":
        st.error("Please enter sample data.")
    else:
        try:
            # Convert input string to numeric array
            data = np.array([float(x.strip()) for x in data_input.split(",")])

            n = len(data)
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)

            # t statistic
            t_cal = (sample_mean - mu) / (sample_std / np.sqrt(n))
            df = n - 1

            # Compute p-value and critical value
            if alternative == "two-sided":
                t_crit_pos = t.ppf(1 - alpha/2, df)
                t_crit_neg = -t_crit_pos
                p_value = 2 * (1 - t.cdf(abs(t_cal), df))

                st.write("t calculated:", t_cal)
                st.write("Critical values:", t_crit_neg, "and", t_crit_pos)

            elif alternative == "greater":
                t_crit = t.ppf(1 - alpha, df)
                p_value = 1 - t.cdf(t_cal, df)

                st.write("t calculated:", t_cal)
                st.write("Critical value:", t_crit)

            else:  # less
                t_crit = t.ppf(alpha, df)
                p_value = t.cdf(t_cal, df)

                st.write("t calculated:", t_cal)
                st.write("Critical value:", t_crit)

            st.write("p-value:", p_value)

            if p_value < alpha:
                st.success("Reject Null Hypothesis")
            else:
                st.info("Fail to Reject Null Hypothesis")

        except:
            st.error("Enter valid numeric values separated by commas.")

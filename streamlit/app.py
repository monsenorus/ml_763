# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# loading files
model = tf.keras.models.load_model("neural_network_model2.keras")
scaler = joblib.load("neural_network_scaler2.pkl")
top_tlds = joblib.load("neural_network_top_tlds2.pkl")

st.title("Phishing Domain Detector")
st.write("Enter the features of a domain to predict whether it is legitimate or phishing.")

# expandable info for users on how to use or interpret the fields below
with st.expander("How to fill in the fields"):
    st.markdown("""
**URL Length**          – Total number of characters in the full URL, including https:// and the path.\n
**Is Domain IP**        – Set to 1 if the domain is a raw IP address (e.g. http://192.168.1.1/...) instead of a name.\n
**Digits in URL**       – How many numeric characters (0–9) appear anywhere in the full URL.\n
**Special Chars**       – Count of special characters in the URL (e.g. -, _, =, ?, &) excluding letters, digits, and slashes.\n
**Is HTTPS**            – Set to 1 if the URL uses HTTPS, 0 if HTTP.\n
**Domain Length**       – Number of characters in the domain name only (e.g. "www.example.com" = 15).\n
**Subdomains**          – Number of dot-separated segments in the domain. "www.example.com" = 1, "mail.store.example.com" = 3. The Top-Level Domain (the last segment of the domain, e.g. ".com" is not to be counted!)\n
**Has Obfuscation**     – Set to 1 if the URL contains obfuscated characters (e.g. %20, hex encoding, or @-tricks).\n
**TLD**                 – The top-level domain extension (e.g. com, org, net). Choose "other" if not in the list.
""")

# creating user inputs
url_length        = st.number_input("URL Length",                       min_value=0,   value=30)
is_domain_ip      = st.selectbox("Is Domain an IP Address?",            [0, 1])
digits_in_url     = st.number_input("Number of Digits in URL",          min_value=0,   value=0)
special_chars     = st.number_input("Number of Special Chars in URL",   min_value=0,   value=1)
is_https          = st.selectbox("Is HTTPS?",                           [1, 0])
domain_length     = st.number_input("Domain Length",                    min_value=0,   value=20)
subdomains        = st.number_input("Number of Subdomains",             min_value=0,   value=2)
has_obfuscation   = st.selectbox("Has Obfuscation?",                    [0, 1])
tld               = st.selectbox("TLD", top_tlds + ["other"])

# feature row
tld_clean = tld if tld in top_tlds else "other"
tld_columns = ["TLD_au", "TLD_co", "TLD_com", "TLD_de", "TLD_io",
               "TLD_net", "TLD_org", "TLD_other", "TLD_ru", "TLD_uk"]

tld_encoding = {col: (col == f"TLD_{tld_clean}") for col in tld_columns}

sample = {
    "URLLength":                   url_length,
    "IsDomainIP":                  is_domain_ip,
    "NoOfDegitsInURL":             digits_in_url,
    "NoOfOtherSpecialCharsInURL":  special_chars,
    "IsHTTPS":                     is_https,
    "DomainLength":                domain_length,
    "NoOfSubDomain":               subdomains,
    "HasObfuscation":              has_obfuscation,
    **tld_encoding
}

sample_df = pd.DataFrame([sample])

# Prediction
if st.button("Predict"):
    sample_scaled = scaler.transform(sample_df)
    prob = model.predict(sample_scaled)[0][0]
    label = "Phishing!" if prob >= 0.5 else "Legitimate"
    if prob >= 0.5:
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")
    st.info(f"Phishing Probability: {prob:.2%}")
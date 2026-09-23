import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sydney Property Estimator")
st.title("Sydney Property Price Estimator")

SUBURB_ORDER = ["Mosman", "Parramatta", "Mount Druitt"]

# load and clean the data, then train the model
@st.cache_data
def load_and_train():
    df = pd.read_csv("housingdatafinal.csv")
    disclosed = df[df["price_disclosed"] == "Yes"].copy()

    # same grouping as Part 3 - lump rare property types into "other"
    disclosed["property_type_grouped"] = disclosed["property_type"].where(
        disclosed["property_type"].isin(["unit", "house"]), "other"
    )

    X = pd.get_dummies(
        disclosed[["suburb", "property_type_grouped", "bedrooms", "bathrooms", "car_spaces"]],
        columns=["suburb", "property_type_grouped"],
    )
    y_log = np.log1p(disclosed["sale_price"].values)

    model = LinearRegression()
    model.fit(X, y_log)

    # training examples per suburb+type combo - used to flag rare combinations
    type_counts = disclosed.groupby(["suburb", "property_type_grouped"]).size()

    return model, X.columns.tolist(), type_counts

model, feature_columns, type_counts = load_and_train()

PROPERTY_TYPE_LABELS = {
    "unit": "Unit / apartment",
    "house": "House",
    "other": "Other (villa, townhouse, semi, retirement)",
}

st.divider()

with st.container(border=True):
    input_col, result_col = st.columns([1, 1.3], gap="large")

    with input_col:
        st.subheader("Property details")
        suburb = st.selectbox("Suburb", SUBURB_ORDER)
        property_type = st.selectbox(
            "Property type",
            ["unit", "house", "other"],
            format_func=lambda x: PROPERTY_TYPE_LABELS[x],
        )

        b1, b2, b3 = st.columns(3)
        bedrooms = b1.number_input("Bedrooms", min_value=0, max_value=8, value=2)
        bathrooms = b2.number_input("Bathrooms", min_value=0, max_value=6, value=1)
        car_spaces = b3.number_input("Car spaces", min_value=0, max_value=6, value=1)

        estimate_clicked = st.button("Estimate price", type="primary", use_container_width=True)

    with result_col:
        st.subheader("Estimate")
        if estimate_clicked:
            row = {col: 0 for col in feature_columns}
            row["bedrooms"] = bedrooms
            row["bathrooms"] = bathrooms
            row["car_spaces"] = car_spaces
            row[f"suburb_{suburb}"] = 1
            row[f"property_type_grouped_{property_type}"] = 1

            input_row = pd.DataFrame([row])[feature_columns]

            predicted_log_price = model.predict(input_row)[0]
            predicted_price = np.expm1(predicted_log_price)

            st.metric("Estimated price", f"${predicted_price:,.0f}")
            st.caption(
                f"{bedrooms} bed, {bathrooms} bath, {car_spaces} car space(s) - "
                f"{PROPERTY_TYPE_LABELS[property_type].lower()} in {suburb}"
            )

            count = type_counts.get((suburb, property_type), 0)
            if count <= 2:
                st.warning(
                    f"Only {count} property of this type in {suburb} appeared in the training data. "
                    f"This estimate is much less reliable than one for a common combination, "
                    f"like a unit in Parramatta or Mount Druitt."
                    if count == 1 else
                    f"Only {count} properties of this type in {suburb} appeared in the training data. "
                    f"This estimate is much less reliable than one for a common combination, "
                    f"like a unit in Parramatta or Mount Druitt."
                )
        else:
            st.info("Fill in the property details and click Estimate price to see a prediction.")
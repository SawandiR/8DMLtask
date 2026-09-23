# Sydney Property Price Estimator

This is a simple Streamlit app I created for the deployment part of my machine learning project.

The app predicts a property sale price using a Linear Regression model based on data collected from three Sydney suburbs: Mosman, Parramatta and Mount Druitt.

## About the project

The dataset contains 101 sold properties, and 85 of them had a disclosed sale price that could be used for modelling.

The model uses:

- Suburb
- Property type
- Bedrooms
- Bathrooms
- Car spaces

Property types are grouped into unit, house and other.

The model was trained using the log of sale price, and the prediction is converted back into dollars before being shown in the app.

## Files

- `app.py` - Streamlit application
- `housingdatafinal.csv` - property dataset
- `requirements.txt` - required Python packages
- `.streamlit/config.toml` - app theme settings

## How to run the app

First install the required packages:

```bash
pip install -r requirements.txt

streamlit run app.py
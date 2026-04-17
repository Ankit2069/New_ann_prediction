# ChurnGuard Pro

An advanced **Machine Learning web application** that intelligently predicts customer churn risk using a sophisticated Artificial Neural Network (ANN). Features a modern dark-themed interface with gradient effects and real-time predictions.

Built with Streamlit | TensorFlow | Scikit-learn | Modern UI/UX

---

## Features

* **ChurnGuard Pro** - Advanced customer retention intelligence system
* Predict customer churn probability with high accuracy
* Displays risk percentage (Churn vs Retention) with visual progress indicator
* Modern dark-themed dashboard with gradient UI elements
* Enhanced typography using Poppins and Inter fonts
* Real-time predictions using trained deep neural network
* Beautiful purple-to-magenta color scheme

---

## How It Works

The model is trained on customer data and uses:

* Gender encoding
* Geography one-hot encoding
* Feature scaling (StandardScaler)
* Artificial Neural Network (ANN)

---

## Project Structure

```
.
├── app.py
├── model.ipynb
├── requirements.txt
├── artifacts/
│   ├── churn_ann_model.keras
│   ├── scaler.pkl
│   └── feature_columns.json
├── Artificial_Neural_Network_Case_Study_data.csv
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Ankit2069/New_ann_prediction.git
cd New_ann_prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Train the Model (Optional)

If you want to retrain the model:

1. Open `model.ipynb`
2. Run all cells
3. Ensure artifacts are generated inside `/artifacts`

---

## Run the App Locally

```
streamlit run app.py
```

---

## Deployment

You can run this app locally on your machine:

```
streamlit run app.py
```

The app will be available at `https://newannprediction-asujtvwdjem5aqhg74oatg.streamlit.app`

### Optional: Streamlit Community Cloud
This app can also be deployed using **Streamlit Community Cloud** for worldwide access.

---

## Notes

* Model predicts churn probability based on comprehensive customer data
* **Churn Risk Threshold:**
  * ≥ 50% → **High Risk** (Red-pink indicator)
  * < 50% → **Low Risk** (Retained customer)
* Ensure all artifacts exist in `/artifacts` directory before running
* The app features a dark gradient background for a modern aesthetic
* Supports customer inputs: Credit Score, Age, Balance, Salary, Tenure, Products, Card Status, Activity Status, Location, and Gender

## UI/UX Updates

* **Dark Theme:** Deep blue-purple gradient background
* **Typography:** Premium fonts (Poppins & Inter) for better readability
* **Color Scheme:** Vibrant purple-to-magenta gradients
* **Interactive Elements:** Smooth hover transitions and enhanced shadows
* **Responsive Design:** Optimized for desktop and tablet viewing

---

## Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies
- TensorFlow CPU (or GPU if available)


---

## Support

If you like this project, give it a star on GitHub!

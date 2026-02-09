# K-Means Customer Segmentation

## 🎯 Overview
This project implements K-Means clustering to segment mall customers into 5 groups based on their annual income and spending scores. It uses **Streamlit** for an interactive web interface.

## 📋 What's Fixed
- ✅ **Pickle Error**: Changed model loading from `pickle` to `joblib` (compatible with how models are saved)
- ✅ **Streamlit Integration**: Converted Flask app to Streamlit for better interactivity
- ✅ **Path Issues**: Fixed hardcoded paths in notebook

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
chmod +x setup.sh
./setup.sh
```

Or manually install:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 2: Generate Models
First, run the Jupyter notebook to train and save the models:
```bash
jupyter notebook "k _ Means clusture.ipynb"
```

Or if you prefer command line:
```bash
python3 -c "
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('Mall_Customers.csv')

# Prepare data
x = df[['Annual Income (k\$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(x_scaled)

# Save models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'means_model.pkl')
print('✅ Models saved successfully!')
"
```

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at **http://localhost:8501**

## 🎨 Features
- 🎯 Interactive sliders for Annual Income and Spending Score
- 🔮 Real-time cluster prediction
- 📊 Cluster descriptions and insights
- 🎭 Color-coded results for easy understanding

## 📊 Customer Segments

| Cluster | Description | Characteristics |
|---------|-------------|-----------------|
| 0 | Budget Conscious | Low income, low spending |
| 1 | Shop Lovers | Low income, high spending |
| 2 | Standard Customers | Average income, average spending |
| 3 | Savers | High income, low spending |
| 4 | Premium Customers | High income, high spending |

## 📁 File Structure
```
K_Means Cluster/
├── streamlit_app.py          # Main Streamlit application
├── app.py                     # Original Flask app (deprecated)
├── k _ Means clusture.ipynb   # Jupyter notebook with training
├── Mall_Customers.csv         # Dataset
├── setup.sh                   # Setup script
├── README.md                  # This file
├── scaler.pkl                 # Trained StandardScaler (generated)
└── means_model.pkl            # Trained KMeans model (generated)
```

## 🔧 Troubleshooting

### Error: "Model files not found"
- Run the notebook first: `jupyter notebook "k _ Means clusture.ipynb"`
- Execute all cells to generate `means_model.pkl` and `scaler.pkl`

### Error: "ModuleNotFoundError"
- Run the setup script: `./setup.sh`
- Or install manually: `pip install streamlit scikit-learn joblib`

### Port Already in Use
- Streamlit uses port 8501 by default
- Run on a different port: `streamlit run streamlit_app.py --server.port 8502`

## 🌐 Accessing Localhost

**Local Machine:**
- Open browser and go to: `http://localhost:8501`

**From Another Machine on Network:**
- Find your IP address:
  - Mac: `ifconfig | grep "inet " | grep -v 127.0.0.1`
  - Linux: `hostname -I`
  - Windows: `ipconfig` (look for IPv4 Address)
- Use: `http://<your-ip>:8501`

## 📝 API Reference (if using Flask instead)

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "age": 35,
  "income": 60,
  "spending": 75
}
```

**Response**:
```json
{
  "cluster": 4
}
```

## 🎓 Running the Notebook Cells

If you want to run the notebook programmatically:

```bash
jupyter nbconvert --to notebook --execute "k _ Means clusture.ipynb"
```

## 💡 Tips
- Always run the notebook first to generate models
- Use Streamlit for interactive exploration
- Models are cached for fast predictions
- Adjust input ranges in code if needed

## 📞 Need Help?
- Check the troubleshooting section
- Ensure all dependencies are installed
- Verify CSV file exists: `Mall_Customers.csv`
- Check that models are generated: `ls *.pkl`

---
**Happy Clustering! 🎯**

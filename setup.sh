#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "📦 Installing required packages..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install required libraries
python3 -m pip install \
    streamlit \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    joblib

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run the Jupyter notebook to train and save the models:"
echo "   jupyter notebook 'k _ Means clusture.ipynb'"
echo ""
echo "2. Then run the Streamlit app:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "The app will be available at http://localhost:8501"

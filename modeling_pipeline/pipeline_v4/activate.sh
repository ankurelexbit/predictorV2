#!/bin/bash
# Activate V4 virtual environment

source venv/bin/activate
echo "✅ V4 virtual environment activated!"
echo ""
echo "Installed packages:"
pip list | grep -E "(pandas|numpy|ijson|scikit-learn|xgboost|matplotlib|seaborn)"
echo ""
echo "Ready to run V4 pipeline scripts!"

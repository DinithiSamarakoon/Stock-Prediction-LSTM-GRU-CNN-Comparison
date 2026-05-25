# 📈 Apple Stock Price Prediction using LSTM, GRU, and CNN+LSTM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements and compares three different deep learning architectures for predicting Apple Inc. (AAPL) stock closing prices:

- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit) 
- **CNN+LSTM** (Hybrid Convolutional + LSTM)

The models are trained on historical stock data from 2015-2024 and evaluated using multiple metrics including RMSE, MAE, MAPE, and R² Score.

## ✨ Key Features

- ✅ **No Data Leakage** - Proper train/test split before scaling
- ✅ **Multiple Models** - Compare 3 different architectures
- ✅ **Comprehensive Metrics** - RMSE, MAE, MAPE, R² Score
- ✅ **Visualizations** - Prediction plots, loss curves, error distributions
- ✅ **Early Stopping** - Prevents overfitting
- ✅ **Learning Rate Scheduling** - Adaptive learning rate
- ✅ **Next Day Prediction** - Forecast future prices
- ✅ **Model Saving** - Export models for future use

## 📊 Results Summary

| Model | RMSE | MAE | MAPE | R² Score |
|-------|------|-----|------|----------|
| **GRU** | **4.11** | **3.21** | **1.64%** | **0.9714** |
| LSTM | 7.34 | 5.64 | 2.79% | 0.9088 |
| CNN+LSTM | 8.85 | 7.07 | 3.50% | 0.8674 |

### 🏆 Best Model: **GRU**
- **97.14%** R² Score indicates excellent fit
- **1.64%** MAPE shows high prediction accuracy
- Only **37,889** parameters (more efficient than LSTM)

## 🏗️ Model Architectures

### LSTM Model

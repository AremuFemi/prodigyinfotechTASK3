# Task 3: Decision Tree Classification Pipeline

## Project Overview
This project builds a Decision Tree Classifier to predict customer conversion outcomes using bank marketing data. The model maps demographic features and historical interaction signals to anticipate purchase behavior.

## Engineering Strategies & Optimizations
1. **Data Leakage Mitigation:** Dropped the `duration` column entirely. In real-world production setups, call durations are unknown prior to a campaign interaction, making its inclusion unrealistic for accurate forward-looking predictions.
2. **Class Imbalance Architecture:** Utilized `class_weight='balanced'` alongside stratified splitting to prevent the model from leaning too heavily toward the majority 'no' class.
3. **Hyperparameter Regulation:** Capped tree complexity at `max_depth=5` to improve model generalization and prevent overfitting on training noise.

## 📊 View Deployment Live
 `https://aremufemi.github.io/prodigyinfotechTASK3/`

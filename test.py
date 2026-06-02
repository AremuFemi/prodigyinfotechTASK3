import pandas as pd
import numpy as np

from task3 import X_encoded, clf

# 1. Create an empty dataframe using the columns from your training data
prediction_template = pd.DataFrame(0, index=[0], columns=X_encoded.columns)

# 2. Update the template with your specific customer data
# Only set the values that are '1' or 'True' for this specific customer
prediction_template['age'] = 54
prediction_template['marital_married'] = 0
prediction_template['job_management'] = 1
prediction_template['housing_yes'] = 1
prediction_template['contact_unknown'] = 0

# 3. Predict
prediction = clf.predict(prediction_template)

# 4. Result
result = "NO" if prediction[0] == 1 else "YES"
print(f"Prediction: Will the customer subscribe? {result}")
# Add new observation and predict the result

"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

import numpy as np

new_prediction = np.array([[600, 0.0, 0, 40, 3, 60000, 2, 1, 1, 50000]])

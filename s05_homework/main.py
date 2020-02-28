# Adding new prediction on the fly

from s04_building_ann.main import data_processing, ann
from s05_homework.c033_add_new_prediction import new_prediction

new_prediction = data_processing.scale(new_prediction)  # Remember to scale the data
new_result = ann.predict(new_prediction)  # Added new prediction
new_result = (new_result > 0.5)

# Examine the new_result cm above

####################################### Load all the pickled files in data folder #######################################

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the data folder path
data_folder = '../../data'

# List all the files in the data folder with .pkl extension
files = [file for file in os.listdir(data_folder) if file.endswith('.pkl')]
print(files)

# Load the county tensor data
county_data = pd.read_pickle(os.path.join(data_folder, 'county_tensor.pkl'))

# Display the data type and shape of the loaded data
print(type(county_data))
print(county_data.shape)

# Remove the code that loads feat_name.pkl
with open(os.path.join(data_folder, 'feat_name.pkl'), 'rb') as f:
    feat_names = pickle.load(f)

# Use the provided feat_name dictionary
# feat_names = {
#     'hospitalization': ['inbeds', 'inbeds_covid', 'icu', 'icu_covid'],
#     'vaccination': ['1st', '2nd', 'bst', 'Pfizer_1', 'Pfizer_2', 'Pfizer_b', 'Moderna_1', 'Moderna_2', 'Moderna_b', 
#                     'Johnson_1', 'Johnson_b', 'PfizerTS_1', 'PfizerTS_2', 'PfizerTS_b', 'PfizerTS10_1', 'PfizerTS10_2'],
#     'claim': ['hospitalization', 'n_visits', 'age_cnt', 'Cerebrovascular Disease', 'Chronic Pulmonary Disease', 
#               'Congestive Heart Failure', 'Dementia', 'Diabetes without chronic complication', 'HIV', 
#               'Hemiplegia or Paraplegia', 'Hypertension', 'Immunodeficiency', 'Liver Disease', 'Malignancy', 
#               'Metastatic Solid Tumor', 'Myocardial Infarction', 'Obesity', 'Peptic Ulcer Disease', 
#               'Peripheral Vascular Disease', 'Renal'],
#     'county': ['pop', '0_17', '18_64', '65p', 'Black', 'White', 'Asian', 'Hispanic', 'Not_Hispanic', 'Physicians', 
#                'Hospitals', 'ICU Beds', 'Income', 'Unemployment_rate'],
#     'date': ['2020-08-01', '2020-08-02', '2020-08-03', '2020-08-04', '2020-08-05', '2020-08-06',
#              # ... more dates ...
#              '2022-05-01']
# }

# Display the loaded feature names
feat_names

# Load the hospitalisation data
hospital_data = pd.read_pickle(os.path.join(data_folder, 'hospitalizations.pkl'))

# Visualize the hospitalisation data
plt.figure(figsize=(10, 6))
plt.imshow(hospital_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Hospitalisations')
plt.title('Hospitalisation Data')
plt.xlabel('Time')
plt.ylabel('Regions')
plt.show()

# visualize the hospitalisation data for the first 10 regions
plt.figure(figsize=(10, 6))
plt.imshow(hospital_data[:10], aspect='auto', cmap='viridis')
plt.colorbar(label='Hospitalisations')
plt.title('Hospitalisation Data for First 10 Regions')
plt.xlabel('Time')
plt.ylabel('Regions')
plt.show()


# what is the length of the hospital_data for 1 region
len(hospital_data[0])
print(f"Length of hospital_data for 1 region: {len(hospital_data[0])} days")

# load the distance matrix data
distance_matrix = pd.read_pickle(os.path.join(data_folder, 'distance_mat.pkl'))
print(distance_matrix.shape)

# Extract region names from feat_names
region_names = feat_names['county']

# Display the distance matrix with improved visualization
plt.figure(figsize=(12, 8))
plt.imshow(distance_matrix, cmap='plasma')
plt.colorbar(label='Distance')
plt.title('Distance Matrix')
plt.xticks(range(len(region_names)), region_names, rotation=90)
plt.yticks(range(len(region_names)), region_names)
plt.show()

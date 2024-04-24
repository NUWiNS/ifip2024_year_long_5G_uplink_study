import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast  # Import the ast module
from matplotlib.pyplot import figure

# Read the .csv file

data = (
    pd.read_csv("Germany_rsrp.csv")
) 


data['Info'] = data['Info'].apply(lambda x: x.replace('nan', 'np.nan'))

# Now use eval
data['Info'] = data['Info'].apply(eval)


#Each entry has 1 key (technology) and 4 values (latency, tput, rsrp and ssrsrp)
#Get the rsrp (LTE) and Ssrsrp (5G) lists


def create_copies(row):
    copies = []
    for key_tuple, value in row['Info'].items():
        new_row = row.copy()
        new_row['key1'] = key_tuple
        new_row['value3'] = value['MeanRsrp']
        new_row['value4'] = value['MeanSsrsrp']
        copies.append(new_row)
    return copies

# Apply the function to create copies and explode the list of copies into separate rows
data_rsrp = pd.DataFrame(data.apply(create_copies, axis=1).explode().tolist())

# Drop the original 'Info' column
data_rsrp = data_rsrp.drop(columns=['Info'])

data_rsrp = data_rsrp.reset_index()


data_rsrp=data_rsrp.rename(columns={"key1":"Technology", "value3":"MeanRsrp","value4":"MeanSsrsrp"})


### 
# Plot RSRP wrt Technology


figure(figsize=(8, 6), dpi=80)
all_values = [value for sublist in data_rsrp['MeanRsrp'] for value in sublist]
all_values_5g = [value for sublist in data_rsrp['MeanSsrsrp'] for value in sublist]

# Flatten the list of lists into a single list of values
flat_values = [value for value in all_values if value != []]
flat_values_5g = [value for value in all_values_5g if value != []]

# Calculate the CDF of the combined values
sorted_values = np.sort(flat_values)
sorted_values_5g = np.sort(flat_values_5g)
cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
cdf_5g = np.arange(1, len(sorted_values_5g) + 1) / len(sorted_values_5g)

plt.plot(sorted_values, cdf, label='LTE', color ='tab:blue')
plt.plot(sorted_values_5g, cdf_5g, label='5G', color='tab:orange')

# Plot the CDF
plt.plot(sorted_values, cdf)
plt.xlabel('RSRP')
plt.ylabel('CDF')
plt.xlim([-130,-50])
#plt.title('CDF of MeanRsrp Values')
plt.grid(True)
plt.legend()
plt.savefig("Plots/Germany_RSRP.png", format="png", dpi=300)
plt.show()


### 
# Plot RSRP wrt Mobility for LTE

all_values_s = [value for sublist in data_rsrp[data_rsrp.Type=="Static"].MeanRsrp for value in sublist]
all_values_w = [value for sublist in data_rsrp[data_rsrp.Type=="Walking"].MeanRsrp for value in sublist]
all_values_d = [value for sublist in data_rsrp[data_rsrp.Type=="Driving"].MeanRsrp for value in sublist]

# Flatten the list of lists into a single list of values
flat_values_s = [value for value in all_values_s if value != []]
flat_values_w = [value for value in all_values_w if value != []]
flat_values_d = [value for value in all_values_d if value != []]

# Calculate the CDF of the combined values
sorted_values_s = np.sort(flat_values_s)
sorted_values_w = np.sort(flat_values_w)
sorted_values_d = np.sort(flat_values_d)

cdf_s = np.arange(1, len(sorted_values_s) + 1) / len(sorted_values_s)
cdf_w = np.arange(1, len(sorted_values_w) + 1) / len(sorted_values_w)
cdf_d = np.arange(1, len(sorted_values_d) + 1) / len(sorted_values_d)

plt.plot(sorted_values_s, cdf_s, label='Static', color ='tab:blue')
plt.plot(sorted_values_w, cdf_w, label='Walking', color ='tab:orange')
plt.plot(sorted_values_d, cdf_d, label='Driving', color ='tab:green')

# Plot the CDF

plt.xlabel('RSRP')
plt.ylabel('CDF')
plt.xlim([-130,-50])
#plt.title('CDF of MeanRsrp Values')
plt.grid(True)
plt.legend()
plt.savefig("Plots/Germany_RSRP_Mobility_LTE.png", format="png", dpi=300)
plt.show()


### 
# Plot RSRP wrt Mobility for 5G

all_values_s = [value for sublist in data_rsrp[data_rsrp.Type=="Static"].MeanSsrsrp for value in sublist]
all_values_w = [value for sublist in data_rsrp[data_rsrp.Type=="Walking"].MeanSsrsrp for value in sublist]
all_values_d = [value for sublist in data_rsrp[data_rsrp.Type=="Driving"].MeanSsrsrp for value in sublist]

# Flatten the list of lists into a single list of values
flat_values_s = [value for value in all_values_s if value != []]
flat_values_w = [value for value in all_values_w if value != []]
flat_values_d = [value for value in all_values_d if value != []]

# Calculate the CDF of the combined values
sorted_values_s = np.sort(flat_values_s)
sorted_values_w = np.sort(flat_values_w)
sorted_values_d = np.sort(flat_values_d)

cdf_s = np.arange(1, len(sorted_values_s) + 1) / len(sorted_values_s)
cdf_w = np.arange(1, len(sorted_values_w) + 1) / len(sorted_values_w)
cdf_d = np.arange(1, len(sorted_values_d) + 1) / len(sorted_values_d)

plt.plot(sorted_values_s, cdf_s, label='Static', color ='tab:blue')
plt.plot(sorted_values_w, cdf_w, label='Walking', color ='tab:orange')
plt.plot(sorted_values_d, cdf_d, label='Driving', color ='tab:green')

# Plot the CDF

plt.xlabel('RSRP')
plt.ylabel('CDF')
plt.xlim([-130,-50])
#plt.title('CDF of MeanRsrp Values')
plt.grid(True)
# plt.legend()
plt.savefig("Plots/Germany_ssrsrp_Mobility_5G.png", format="png", dpi=300)
plt.show()

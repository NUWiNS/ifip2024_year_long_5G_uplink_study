import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure

data = (
    pd.read_csv("Norway_rsrp.csv")

) 

def literal_eval_with_nan(s):
    try:
        return ast.literal_eval(s.replace('nan', 'None'))
    except (ValueError, SyntaxError):
        return None

def separate_and_expand_entries(data):
    # Create an empty list to store the separated and expanded rows
    expanded_rows = []

    # Iterate through each row in the DataFrame
    for _, row in data.iterrows():
        # Convert the string representation of the dictionary to an actual dictionary
        mean_data_dict_str = row['Info']
        mean_data_dict = literal_eval_with_nan(mean_data_dict_str)

        # Check if the value is a dictionary
        if isinstance(mean_data_dict, dict):
            # Iterate through the entries in the dictionary
            for entry_key, entry_value in mean_data_dict.items():
                # Create a copy of the row
                new_row = row.copy()

                # Extract information from the entry key
                new_row['Technology'] = entry_key

                # Extract information from the entry value
                new_row['Mean_Latency'] = entry_value.get('MeanLatency')
                new_row['Mean_Throughput'] = entry_value.get('MeanThroughput')
                new_row['MeanRsrp'] = entry_value.get('MeanRsrp')
                new_row['MeanSsrsrp'] = entry_value.get('MeanSsrsrp')
                new_row['Counter_tput'] = entry_value.get('Counter_tput')
                new_row['Counter_latency'] = entry_value.get('Counter_latency')

                # Append the expanded row to the list
                expanded_rows.append(new_row)
        else:
            # If the value is not a dictionary or conversion fails, simply append the original row
            expanded_rows.append(row)

    # Create a new DataFrame from the list of expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df


expanded_data = separate_and_expand_entries(data)

all_values = [value for sublist in expanded_data['MeanRsrp'] for value in sublist]
all_values_5g = [value for sublist in expanded_data['MeanSsrsrp'] for value in sublist]

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
plt.grid(True)
plt.savefig("Plots/Norway_rsrp.png", format="png", dpi=300)



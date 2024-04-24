import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure


data = (
    pd.read_csv("Italy.csv")

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
                new_row['Technology'], new_row['Cell_ID'], new_row['Physical_Cell_ID'] = entry_key

                # Extract information from the entry value
                new_row['Mean_Latency'] = entry_value.get('MeanLatency')
                new_row['Mean_Throughput'] = entry_value.get('MeanThroughput')

                # Append the expanded row to the list
                expanded_rows.append(new_row)
        else:
            # If the value is not a dictionary or conversion fails, simply append the original row
            expanded_rows.append(row)

    # Create a new DataFrame from the list of expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

expanded_data = separate_and_expand_entries(data)


###
# Plot Latency CDF


fig, ax = plt.subplots()
rtt_LTE = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency) & (expanded_data.Technology=="LTE") )  ], x="Mean_Latency")
rtt_5G = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency) & (expanded_data.Technology=="5G") )  ], x="Mean_Latency")

rtt_LTE.set(xlim=(10, 200))


ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid()  

plt.savefig("Plots/Italy_Latency.png", format="png",dpi=300)


###
# Plot Throughput CDF


fig, ax = plt.subplots()
thput_LTE = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="LTE") )  ], x="Mean_Throughput")
thput_5G = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="5G") )  ], x="Mean_Throughput")

thput_LTE.set(xlim=(0, 200))

ax.set_xlabel("Throughput [Mbps]")
ax.set_ylabel("CDF")
plt.grid()  

plt.savefig("Plots/Italy_Throughput.png", format="png", dpi=300)




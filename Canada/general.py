import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure

data = (
    pd.read_csv("Canada.csv")    
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
# Extract ARFCN values



def extract_values(row):
    try:
        data_dict = eval(row['cell_info_data'])
        return pd.Series(data_dict)
    except (SyntaxError, TypeError):
        return pd.Series({'ARFCN': None, 'EARFCN': None})
    
expanded_data = pd.concat([expanded_data, expanded_data.apply(extract_values, axis=1)], axis=1)
expanded_data = expanded_data.drop('cell_info_data', axis=1)


###
# Plot Latency CDF


fig, ax = plt.subplots()


rtt_LTE = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency) & (expanded_data.Technology=="LTE") )  ], x="Mean_Latency")
rtt_5G_tot = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency) & (expanded_data.Technology=="5G") )  ], x="Mean_Latency", color="tab:orange")
rtt_5G_low = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency)& (expanded_data.Technology=="5G") ) &(((expanded_data.ARFCN == '126400') | (expanded_data.ARFCN == '126490')) & (pd.notna(expanded_data.ARFCN)))], x="Mean_Latency", color="purple")
rtt_5G_mid = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Latency) & (expanded_data.Technology=="5G") ) &((expanded_data.ARFCN != '126400') & (expanded_data.ARFCN != '126490') & (pd.notna(expanded_data.ARFCN)))], x="Mean_Latency", color="green")

rtt_LTE.set(xlim=(10, 200))

ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid() 


plt.savefig("Plots/Canada_Latency.png", format="png", dpi=300)

###
# Plot Throughput CDF


fig, ax = plt.subplots()

tput_LTE = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="LTE") )  ], x="Mean_Throughput")
tput_5G_tot = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="5G") )  ], x="Mean_Throughput", color='tab:orange')
tput_5G_low = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput)& (expanded_data.Technology=="5G") ) &(((expanded_data.ARFCN == '126400') | (expanded_data.ARFCN == '126490')) & (pd.notna(expanded_data.ARFCN)))], x="Mean_Throughput", color="purple")
tput_5G_mid = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="5G") ) &((expanded_data.ARFCN != '126400') & (expanded_data.ARFCN != '126490') & (pd.notna(expanded_data.ARFCN)))], x="Mean_Throughput", color="green")


ax.set_ylabel("CDF")
rtt_LTE.set(xlim=(0, 200))
plt.grid() 
ax.set_xlabel("Throughput [Mbps]")

plt.savefig("Plots/Canada_Throughput.png", format="png", dpi=300)

plt.show()


###
# Plot Latency wrt Mobility CDF for 5G

expanded_data['Mobility_Mode_5G'] = expanded_data['Type']
expanded_data['Mobility_Mode_LTE'] = expanded_data['Type']


fig, ax = plt.subplots()
rtt_5g = sns.ecdfplot(
    data=expanded_data[pd.notna(expanded_data['Mean_Latency']) & (expanded_data['Technology'] == "5G")],
    x="Mean_Latency", hue="Type", hue_order=["Static", "Walking", "Driving"]  
)
rtt_5g.set(xlim=(20, 100))
rtt_5g.set_xlabel("RTT (5G) [ms]")
rtt_5g.get_legend().set_title("")
rtt_5g.set_ylabel("CDF")
plt.grid()  
rtt_5g.get_legend().remove()


plt.savefig("Plots/Canada_Mobility_5G.png", format="png", dpi=300)


###
# Plot Latency wrt Mobility CDF for LTE


# Plot for LTE
fig, ax = plt.subplots()
rtt_lte = sns.ecdfplot(
    data=expanded_data[pd.notna(expanded_data['Mean_Latency']) & (expanded_data['Technology'] == "LTE")],
    x="Mean_Latency", hue="Type", hue_order=["Static", "Walking", "Driving"]  
)
rtt_lte.set(xlim=(20, 100))
rtt_lte.set_xlabel("RTT (LTE) [ms]")
rtt_lte.get_legend().set_title("")
rtt_lte.set_ylabel("CDF")
plt.grid()  
rtt_lte.get_legend().remove()


plt.savefig("Plots/Canada_Mobility_LTE.png", format="png", dpi=300)

###
# Plot Throughput wrt Mobility CDF for 5G


fig, ax = plt.subplots()
rtt_5g = sns.ecdfplot(
    data=expanded_data[pd.notna(expanded_data['Mean_Throughput']) & (expanded_data['Technology'] == "5G")],
    x="Mean_Throughput", hue="Type", hue_order=["Static", "Walking", "Driving"]  # Specify the desired order
)
rtt_5g.set(xlim=(0, 150))
rtt_5g.set_xlabel("Throughput (5G) [Mbps]")
rtt_5g.get_legend().set_title("")
rtt_5g.set_ylabel("CDF")
plt.grid()  
rtt_5g.get_legend().remove()


plt.savefig("Plots/Canada_Mobility_5G_tput.png", format="png", dpi=300)

###
# Plot Throughput wrt Mobility CDF for LTE


fig, ax = plt.subplots()
rtt_lte = sns.ecdfplot(
    data=expanded_data[pd.notna(expanded_data['Mean_Throughput']) & (expanded_data['Technology'] == "LTE")],
    x="Mean_Throughput", hue="Type", hue_order=["Static", "Walking", "Driving"]  # Specify the desired order
)
rtt_lte.set(xlim=(0, 150))
rtt_lte.set_xlabel("Throughput (LTE) [Mbps]")
rtt_lte.get_legend().set_title("")
rtt_lte.set_ylabel("CDF")
plt.grid() 
rtt_lte.get_legend().remove()

plt.savefig("Plots/Canada_Mobility_LTE_tput.png", format="png", dpi=300)
plt.show()


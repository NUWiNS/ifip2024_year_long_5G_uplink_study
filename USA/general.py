import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure

data = pd.read_csv("USA.csv")    

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

def extract_values(row):
    try:
        data_dict = eval(row['cell_info_data'])
        return pd.Series(data_dict)
    except (SyntaxError, TypeError):
        return pd.Series({'ARFCN': None, 'EARFCN': None})

# Apply the custom function and concatenate the result with the original DataFrame
expanded_data = pd.concat([expanded_data, expanded_data.apply(extract_values, axis=1)], axis=1)

# Drop the original 'cell_info_data' column
expanded_data = expanded_data.drop('cell_info_data', axis=1)



###
# Plot Latency CDF CA

fig, ax = plt.subplots()

expanded_data_ca=expanded_data[expanded_data.State=="CA"]


rtt_LTE = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca["Mean_Latency"])) & (expanded_data_ca.Technology=="LTE")], x="Mean_Latency")
rtt_5G_tot = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca["Mean_Latency"])) & (expanded_data_ca.Technology=="5G")], x="Mean_Latency", color="tab:orange")
rtt_5G_low = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca.Mean_Latency)) & (expanded_data_ca.Technology=="5G") &(((expanded_data_ca.ARFCN == '125900') | (expanded_data_ca.ARFCN == '125570') | (expanded_data_ca.ARFCN == '174800')| (expanded_data_ca.ARFCN == '125400')))], x="Mean_Latency", color="purple")
rtt_5G_mid = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca.Mean_Latency)) & (expanded_data_ca.Technology=="5G") &((expanded_data_ca.ARFCN == '529998') | (expanded_data_ca.ARFCN == '501390') | (expanded_data_ca.ARFCN == '528696') | (expanded_data_ca.ARFCN == '520110') | (expanded_data_ca.ARFCN == '524190')) |  (((pd.notna(expanded_data_ca["Mean_Latency"])))&(expanded_data_ca.Technology=="5G")&(expanded_data_ca['ARFCN'].isna()) & ((expanded_data_ca.OverrideNetworkType=="Unknown")|(expanded_data_ca.OverrideNetworkType=="GHz)")))], x="Mean_Latency", color="green")


# # Set x-axis label
ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid() 

rtt_LTE.set(xlim=(0, 200))

plt.savefig("Plots/USA_Latency_ca.png", format="png", dpi=300)


###
# Plot Throughput CDF CA

fig, ax = plt.subplots()
expanded_data_ca=expanded_data[expanded_data.State=="CA"]

rtt_LTE = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca["Mean_Throughput"])) & (expanded_data_ca.Technology=="LTE")], x="Mean_Throughput")
rtt_5G_tot = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca["Mean_Throughput"])) & (expanded_data_ca.Technology=="5G")], x="Mean_Throughput", color="tab:orange")
rtt_5G_low = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca.Mean_Throughput)) & (expanded_data_ca.Technology=="5G") &(((expanded_data_ca.ARFCN == '125900') | (expanded_data_ca.ARFCN == '125570') | (expanded_data_ca.ARFCN == '174800')| (expanded_data_ca.ARFCN == '125400')))], x="Mean_Throughput", color="purple")
rtt_5G_mid = sns.ecdfplot(data=expanded_data_ca[(pd.notna(expanded_data_ca.Mean_Throughput)) & (expanded_data_ca.Technology=="5G") &((expanded_data_ca.ARFCN == '529998') | (expanded_data_ca.ARFCN == '501390') | (expanded_data_ca.ARFCN == '528696') | (expanded_data_ca.ARFCN == '520110') | (expanded_data_ca.ARFCN == '524190')) |  (((pd.notna(expanded_data_ca["Mean_Throughput"])))&(expanded_data_ca.Technology=="5G")&(expanded_data_ca['ARFCN'].isna()) & ((expanded_data_ca.OverrideNetworkType=="Unknown")|(expanded_data_ca.OverrideNetworkType=="GHz)")))], x="Mean_Throughput", color="green")




# # Set x-axis label
ax.set_xlabel("Throughput [Mbps]")
ax.set_ylabel("CDF")
plt.grid()
rtt_LTE.set(xlim=(0, 200))

plt.savefig("Plots/USA_Throughput_ca.png", format="png", dpi=300)



###
# Plot Latency CDF MA

fig, ax = plt.subplots()
expanded_data_ma=expanded_data[expanded_data.State=="MA"]

# expanded_data.ARFCN

rtt_LTE = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma["Mean_Latency"])) & (expanded_data_ma.Technology=="LTE")], x="Mean_Latency")
rtt_LTE_tot = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma["Mean_Latency"])) & (expanded_data_ma.Technology=="5G")], x="Mean_Latency", color="tab:orange")
rtt_5G_low = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma.Mean_Latency)) & (expanded_data_ma.Technology=="5G") &(((expanded_data_ma.ARFCN == '125900') | (expanded_data_ma.ARFCN == '125570') | (expanded_data_ma.ARFCN == '174800')| (expanded_data_ma.ARFCN == '125400'))) ], x="Mean_Latency", color="purple")
rtt_5G_mid = sns.ecdfplot(data=expanded_data_ma[((pd.notna(expanded_data_ma.Mean_Latency) & (expanded_data_ma.Technology=="5G") &((expanded_data_ma.ARFCN == '529998') | (expanded_data_ma.ARFCN == '501390') | (expanded_data_ma.ARFCN == '528696') | (expanded_data_ma.ARFCN == '520110') | (expanded_data_ma.ARFCN == '524190') ) )) | ((expanded_data_ma['ARFCN'].isna()) & ((expanded_data_ma.OverrideNetworkType=="Unknown")|(expanded_data_ma.OverrideNetworkType=="GHz)")) & (expanded_data_ma.Technology=="5G") & (pd.notna(expanded_data_ma.Mean_Latency)))], x="Mean_Latency", color="green")
rtt_5G_high = sns.ecdfplot(data=expanded_data_ma[((pd.notna(expanded_data_ma.Mean_Latency)) & (expanded_data_ma.Technology=="5G")&((expanded_data_ma.ARFCN == '2073333')) ) | ((expanded_data_ma['ARFCN'].isna()) & (expanded_data_ma.OverrideNetworkType=="5G(mmWave)") & (expanded_data_ma.Technology=="5G") & (pd.notna(expanded_data_ma.Mean_Latency)))], x="Mean_Latency", color="tab:red")


ax.legend(labels=["LTE" , "5G", "5G-low" ,"5G-mid" , "5G-high"], loc='lower right')

# # Set x-axis label
ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid() 
rtt_LTE.set(xlim=(10, 200))

plt.savefig("Plots/USA_Latency_ma.png", format="png", dpi=300)


###
# Plot Throughput CDF MA

fig, ax = plt.subplots()
expanded_data_ma=expanded_data[expanded_data.State=="MA"]

# expanded_data.ARFCN

rtt_LTE = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma["Mean_Throughput"])) & (expanded_data_ma.Technology=="LTE")], x="Mean_Throughput")
rtt_5G_tot = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma["Mean_Throughput"])) & (expanded_data_ma.Technology=="5G")], x="Mean_Throughput", color="tab:orange")
rtt_5G_low = sns.ecdfplot(data=expanded_data_ma[(pd.notna(expanded_data_ma.Mean_Throughput)) & (expanded_data_ma.Technology=="5G") &(((expanded_data_ma.ARFCN == '125900') | (expanded_data_ma.ARFCN == '125570') | (expanded_data_ma.ARFCN == '174800')| (expanded_data_ma.ARFCN == '125400'))) ], x="Mean_Throughput",  color="purple")
rtt_5G_mid = sns.ecdfplot(data=expanded_data_ma[((pd.notna(expanded_data_ma.Mean_Throughput) & (expanded_data_ma.Technology=="5G") &((expanded_data_ma.ARFCN == '529998') | (expanded_data_ma.ARFCN == '501390') | (expanded_data_ma.ARFCN == '528696') | (expanded_data_ma.ARFCN == '520110') | (expanded_data_ma.ARFCN == '524190') ) )) | ((expanded_data_ma['ARFCN'].isna()) & ((expanded_data_ma.OverrideNetworkType=="Unknown")|(expanded_data_ma.OverrideNetworkType=="GHz)")) & (expanded_data_ma.Technology=="5G") & (pd.notna(expanded_data_ma.Mean_Throughput)))], x="Mean_Throughput", color="green")
rtt_5G_high = sns.ecdfplot(data=expanded_data_ma[((pd.notna(expanded_data_ma.Mean_Throughput)) & (expanded_data_ma.Technology=="5G")&((expanded_data_ma.ARFCN == '2073333')) ) | ((expanded_data_ma['ARFCN'].isna()) & (expanded_data_ma.OverrideNetworkType=="5G(mmWave)") & (expanded_data_ma.Technology=="5G") & (pd.notna(expanded_data_ma.Mean_Throughput)))], x="Mean_Throughput", color="red")



ax.legend(labels=["LTE" ,"5G", "5G-low" ,"5G-mid" , "5G-high"], loc='lower right')

# # Set x-axis label
ax.set_xlabel("Throughput [Mbps]")
ax.set_ylabel("CDF")
plt.grid()  
rtt_LTE.set(xlim=(0, 200))

plt.savefig("Plots/USA_Throughput_ma.png", format="png", dpi=300)

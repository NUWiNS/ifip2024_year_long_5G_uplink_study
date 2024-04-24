import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure


data = (
    pd.read_csv("Norway.csv")

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

rtt_LTE.set(xlim=(0, 200))

ax.legend(labels=["LTE", "5G" ], loc='lower right')


ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid()
plt.savefig("Plots/Norway_Latency.png", format="png", dpi=300)

###
# Plot Throughput CDF

fig, ax = plt.subplots()
thput_LTE = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="LTE") )  ], x="Mean_Throughput")
thput_5G = sns.ecdfplot(data=expanded_data[(pd.notna(expanded_data.Mean_Throughput) & (expanded_data.Technology=="5G") )  ], x="Mean_Throughput")

thput_LTE.set(xlim=(0, 200))

ax.legend(labels=["LTE", "5G" ], loc='lower right')


ax.set_xlabel("Throughput [Mbps]")
ax.set_ylabel("CDF")
plt.grid() 
plt.savefig("Plots/Norway_Throughput.png", format="png",dpi=300)

###
# Plot Evolution of 5G during the duration of measurements


data = expanded_data.dropna(subset=['Date'])
converted =data['Date'].apply(datetime.fromtimestamp)
time =pd.DataFrame(converted)


time['formatted'] = pd.to_datetime(time['Date'])
time['date'] = time['formatted'].dt.strftime('%Y-%m-%d')


converted_data=[data, time.date]
data_time = pd.concat(converted_data, axis=1)
data_time = data_time.sort_values('Date')
data_time[['Year', 'Month', 'Day']]=data_time['date'].str.split('-', expand=True)



data_time['date'] = pd.to_datetime(data_time['date'], errors='coerce')
data_time['Week_Number'] = data_time['date'].dt.isocalendar().week


plt.figure(figsize=(10, 4))

# Convert 'Week_Number' to float
data_time['Week_Number'] = data_time['Week_Number'].astype(float)

# Group by Year and Week_Number and calculate the mean throughput for 5G
grouped_data1 = data_time[(data_time.Technology=="5G") & (data_time['Mean_Throughput'].notna())].groupby(['Year', 'Week_Number']).Mean_Throughput.mean().reset_index()

# Combine 'Year' and 'Week_Number' to create a new column for x-axis
grouped_data1['Year_Week'] = grouped_data1['Year'] + ' - W' + grouped_data1['Week_Number'].astype(int).astype(str)

# Perform a 3-week moving average with overlap
grouped_data1['Mean_Throughput'] = grouped_data1.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

# Filter for 'Walking' type and calculate mean throughput
grouped_data_walking = data_time[(data_time.Technology=="5G") & (data_time['Mean_Throughput'].notna()) & (data_time['Type']=="Walking")].groupby(['Year', 'Week_Number']).Mean_Throughput.mean().reset_index()
grouped_data_walking['Year_Week'] = grouped_data_walking['Year'] + ' - W' + grouped_data_walking['Week_Number'].astype(int).astype(str)
grouped_data_walking['Mean_Throughput'] = grouped_data_walking.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

# Filter for 'Driving' type and calculate mean throughput
grouped_data_driving = data_time[(data_time.Technology=="5G") & (data_time['Mean_Throughput'].notna()) & (data_time['Type']=="Driving")].groupby(['Year', 'Week_Number']).Mean_Throughput.mean().reset_index()
grouped_data_driving['Year_Week'] = grouped_data_driving['Year'] + ' - W' + grouped_data_driving['Week_Number'].astype(int).astype(str)
grouped_data_driving['Mean_Throughput'] = grouped_data_driving.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())


# Remove the last data point
grouped_data1 = grouped_data1.iloc[:-1]
grouped_data_driving = grouped_data_driving.iloc[:-1]

# Plot the lines for total, walking, and driving throughput
sns.lineplot(x='Year_Week', y='Mean_Throughput', data=grouped_data1, label="Total")
sns.lineplot(x='Year_Week', y='Mean_Throughput', data=grouped_data_walking, label="Walking", color='tab:orange')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Set modified ticks on the x-axis
original_ticks = plt.xticks()[0]
modified_ticks = [tick for i, tick in enumerate(original_ticks) if i % 2 == 0]
plt.xticks(modified_ticks)

plt.xlabel("")
plt.ylabel("Mean Throughput")
plt.tight_layout()
plt.grid()

# Save the plot
plt.savefig("Plots/Norway_LinePlot5G.png", format="png", dpi=300)

# Show the plot
plt.show()



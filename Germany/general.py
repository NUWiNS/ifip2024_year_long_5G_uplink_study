import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast  # Import the ast module
from matplotlib.pyplot import figure

data = (
    pd.read_csv("Germany.csv")
) 


data['Info'] = data['Info'].apply(lambda x: x.replace('nan', 'np.nan'))

# Now use eval
data['Info'] = data['Info'].apply(eval)

def create_copies(row):
    copies = []
    for key_tuple, value in row['Info'].items():
        new_row = row.copy()
        new_row['key1'] = key_tuple[0]
        new_row['key2'] = key_tuple[1]
        new_row['key3'] = key_tuple[2]
        new_row['value1'] = value['MeanLatency']
        new_row['value2'] = value['MeanThroughput']
        copies.append(new_row)
    return copies

# Apply the function to create copies and explode the list of copies into separate rows
data = pd.DataFrame(data.apply(create_copies, axis=1).explode().tolist())

# Drop the original 'Info' column
data = data.drop(columns=['Info'])

data = data.reset_index()
data=data.rename(columns={"key1":"Technology", "key2":"Cell_ID", "key3":"Physical_Cell_ID", "value1":"Mean_Latency", "value2":"Mean_Throughput"})

###
# Plot Latency CDF

fig, ax = plt.subplots()

rtt_LTE = sns.ecdfplot(data=data[(pd.notna(data.Mean_Latency) & (data.Technology=="LTE") )  ], x="Mean_Latency")
rtt_5G = sns.ecdfplot(data=data[(pd.notna(data.Mean_Latency) & (data.Technology=="5G") )  ], x="Mean_Latency")

rtt_LTE.set(xlim=(10, 200))

ax.legend(labels=["LTE" , "5G"], loc='lower right')


ax.set_xlabel("RTT [ms]")
ax.set_ylabel("CDF")
plt.grid()  #just add this


plt.savefig("Plots/Germany_Latency.png", format="png", dpi=300)

###
# Plot Throughput CDF

fig, ax = plt.subplots()
thput_LTE = sns.ecdfplot(data=data[(pd.notna(data.Mean_Throughput) & (data.Technology=="LTE") )  ], x="Mean_Throughput")
thput_5G = sns.ecdfplot(data=data[(pd.notna(data.Mean_Throughput) & (data.Technology=="5G") )  ], x="Mean_Throughput")

thput_LTE.set(xlim=(0, 200))

# ax.legend(labels=["LTE-" + str(round(len(df_copies[(df_copies.Technology=="LTE") & (pd.notna(df_copies.Mean_Throughput))])/len(df_copies[pd.notna(df_copies.Mean_Throughput)]),2)) , "5G-" + str(round(len(df_copies[(df_copies.Technology=="5G") & (pd.notna(df_copies.Mean_Throughput))])/len(df_copies[pd.notna(df_copies.Mean_Throughput)]),2))])
ax.legend(labels=["LTE"  , "5G"], loc='lower right')


ax.set_xlabel("Throughput [Mbps]")
# ax.set_title("Germany")
ax.set_ylabel("CDF")
plt.grid()  #just add this

plt.savefig("Plots/Germany_Throughput.png", format="png", dpi=300)

###
# Plot Throughput wrt Mobility CDF for LTE


fig, ax = plt.subplots()
rtt_lte = sns.ecdfplot(
    data=data[pd.notna(data['Mean_Throughput']) & (data['Technology'] == "LTE")],
    x="Mean_Throughput", hue="Type", hue_order=["Static", "Walking", "Driving"]  # Specify the desired order
)
rtt_lte.set(xlim=(0, 150))
rtt_lte.set_xlabel("Throughput (LTE) [Mbps]")
rtt_lte.get_legend().set_title("")
rtt_lte.set_ylabel("CDF")
plt.grid()  #just add this

legend_lte = ax.get_legend()
ax.add_artist(legend_lte)
legend_lte.set_bbox_to_anchor((1, 0.37))

plt.savefig("Plots/Germany_Mobility_LTE_tput.png", format="png", dpi=300)

###
# Plot Throughput wrt Mobility CDF for 5G

fig, ax = plt.subplots()
rtt_5g = sns.ecdfplot(
    data=data[pd.notna(data['Mean_Throughput']) & (data['Technology'] == "5G")],
    x="Mean_Throughput", hue="Type", hue_order=["Static", "Walking", "Driving"]  # Specify the desired order
)
rtt_5g.set(xlim=(0, 150))
rtt_5g.set_xlabel("Throughput (5G) [Mbps]")
rtt_5g.set_ylabel("CDF")
rtt_5g.get_legend().set_title("")
plt.grid()  #just add this
rtt_5g.get_legend().remove()


plt.savefig("Plots/Germany_Mobility_5G_tput.png", format="png", dpi=300)

###
# Plot Evolution of 5G during the duration of measurements

data = data.dropna(subset=['Date'])
converted =data['Date'].apply(datetime.fromtimestamp)
time =pd.DataFrame(converted)
time['formatted'] = pd.to_datetime(time['Date'])
time['date'] = time['formatted'].dt.strftime('%Y-%m-%d')

# Sort the data according to date

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

# Perform a x-week moving average with overlap
grouped_data1['Mean_Throughput'] = grouped_data1.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

# Filter for 'Walking' type and calculate mean throughput
grouped_data_walking = data_time[(data_time.Technology=="5G") & (data_time['Mean_Throughput'].notna()) & (data_time['Type']=="Walking")].groupby(['Year', 'Week_Number']).Mean_Throughput.mean().reset_index()
grouped_data_walking['Year_Week'] = grouped_data_walking['Year'] + ' - W' + grouped_data_walking['Week_Number'].astype(int).astype(str)
grouped_data_walking['Mean_Throughput'] = grouped_data_walking.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

# Filter for 'Driving' type and calculate mean throughput
grouped_data_driving = data_time[(data_time.Technology=="5G") & (data_time['Mean_Throughput'].notna()) & (data_time['Type']=="Driving")].groupby(['Year', 'Week_Number']).Mean_Throughput.mean().reset_index()
grouped_data_driving['Year_Week'] = grouped_data_driving['Year'] + ' - W' + grouped_data_driving['Week_Number'].astype(int).astype(str)
grouped_data_driving['Mean_Throughput'] = grouped_data_driving.groupby('Year')['Mean_Throughput'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

# Plot the lines for total, walking, and driving throughput
sns.lineplot(x='Year_Week', y='Mean_Throughput', data=grouped_data1)
sns.lineplot(x='Year_Week', y='Mean_Throughput', data=grouped_data_driving, color='tab:green')

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
plt.savefig("Plots/Germany_LinePlot5G.png", format="png", dpi=300)



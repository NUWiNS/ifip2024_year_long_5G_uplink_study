import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast 
from matplotlib.pyplot import figure


data = pd.read_csv("Total.csv")



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


data_germany=expanded_data[((expanded_data.Country=="Deutschland") | (expanded_data.Country=="Germany")) & (expanded_data.City == "Berlin")]
data_italy=expanded_data[(expanded_data.Country=="Italy")]
data_norway=expanded_data[(expanded_data.Country=="Norway")]
data_portugal=expanded_data[(expanded_data.Country=="Portugal")]
data_canada=expanded_data[(expanded_data.Country=="Canada")]
data_spain=expanded_data[(expanded_data.Country=="España")]
data_usa_ma=expanded_data[((expanded_data.Country=="USA") | (expanded_data.Country=="United States")) & (expanded_data.StateUSA == "MA")]
data_usa_ca=expanded_data[((expanded_data.Country=="USA") | (expanded_data.Country=="United States")) & (expanded_data.StateUSA == "CA")]

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

# Define the cities and their corresponding data
cities = ['Berlin', 'Turin', 'Oslo', 'Porto', 'Madrid', 'Vancouver', 'Boston', 'Bay Area']
data_by_city = {
    'Berlin': {
        'LTE': data_germany[(pd.notna(data_germany.Mean_Latency) & (data_germany.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_germany[(pd.notna(data_germany.Mean_Latency) & (data_germany.Technology=="5G"))]["Mean_Latency"]
    },
    'Turin': {
        'LTE': data_italy[(pd.notna(data_italy.Mean_Latency) & (data_italy.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_italy[(pd.notna(data_italy.Mean_Latency) & (data_italy.Technology=="5G"))]["Mean_Latency"]
    },
    'Oslo': {
        'LTE': data_norway[(pd.notna(data_norway.Mean_Latency) & (data_norway.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_norway[(pd.notna(data_norway.Mean_Latency) & (data_norway.Technology=="5G"))]["Mean_Latency"]
    },
    'Porto': {
        'LTE': data_portugal[(pd.notna(data_portugal.Mean_Latency) & (data_portugal.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_portugal[(pd.notna(data_portugal.Mean_Latency) & (data_portugal.Technology=="5G"))]["Mean_Latency"]
    },
    'Madrid': {
        'LTE': data_spain[(pd.notna(data_spain.Mean_Latency) & (data_spain.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_spain[(pd.notna(data_spain.Mean_Latency) & (data_spain.Technology=="5G"))]["Mean_Latency"]
    },
        'Vancouver': {
        'LTE': data_canada[(pd.notna(data_canada.Mean_Latency) & (data_canada.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_canada[(pd.notna(data_canada.Mean_Latency) & (data_canada.Technology=="5G"))]["Mean_Latency"]
    },
    'Boston': {
        'LTE': data_usa_ma[(pd.notna(data_usa_ma.Mean_Latency) & (data_usa_ma.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_usa_ma[(pd.notna(data_usa_ma.Mean_Latency) & (data_usa_ma.Technology=="5G"))]["Mean_Latency"]
    },
    'Bay Area': {
        'LTE': data_usa_ca[(pd.notna(data_usa_ca.Mean_Latency) & (data_usa_ca.Technology=="LTE"))]["Mean_Latency"],
        '5G': data_usa_ca[(pd.notna(data_usa_ca.Mean_Latency) & (data_usa_ca.Technology=="5G"))]["Mean_Latency"]
    }
}

# Create boxplots for each city
fig, ax = plt.subplots(figsize=(12, 6))
box_width = 0.4
colors = {'LTE': 'tab:blue', '5G': 'tab:orange'}
for i, city in enumerate(cities):
    positions = [i - box_width/2, i + box_width/2]
    for tech in ['LTE', '5G']:
        box_data = data_by_city[city][tech]
        ax.boxplot(box_data, positions=[positions[0] if tech == 'LTE' else positions[1]], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=colors[tech], color='black'),medianprops=dict(color='black'))
ax.set_ylim([10, 150])        
ax.set_xticks(np.arange(len(cities)))
ax.set_xticklabels(cities)
# ax.set_xlabel('Cities')
ax.set_ylabel('Latency [ms]')
# ax.set_title('Mean Latency of LTE and 5G in Different Cities')

# Create custom legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=colors[tech], label=tech) for tech in ['LTE', '5G']]
ax.legend(handles=handles)
plt.grid() 

plt.savefig("Plots/Mean_Latency_All.png", format="png", dpi=300)




def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

# Define the cities and their corresponding data
cities = ['Berlin', 'Turin', 'Oslo', 'Porto', 'Madrid', 'Vancouver', 'Boston', 'Bay Area']
data_by_city = {
    'Berlin': {
        'LTE': data_germany[(pd.notna(data_germany.Mean_Throughput) & (data_germany.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_germany[(pd.notna(data_germany.Mean_Throughput) & (data_germany.Technology=="5G"))]["Mean_Throughput"]
    },
    'Turin': {
        'LTE': data_italy[(pd.notna(data_italy.Mean_Throughput) & (data_italy.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_italy[(pd.notna(data_italy.Mean_Throughput) & (data_italy.Technology=="5G"))]["Mean_Throughput"]
    },
    'Oslo': {
        'LTE': data_norway[(pd.notna(data_norway.Mean_Throughput) & (data_norway.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_norway[(pd.notna(data_norway.Mean_Throughput) & (data_norway.Technology=="5G"))]["Mean_Throughput"]
    },
    'Porto': {
        'LTE': data_portugal[(pd.notna(data_portugal.Mean_Throughput) & (data_portugal.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_portugal[(pd.notna(data_portugal.Mean_Throughput) & (data_portugal.Technology=="5G"))]["Mean_Throughput"]
    },
    'Madrid': {
        'LTE': data_spain[(pd.notna(data_spain.Mean_Throughput) & (data_spain.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_spain[(pd.notna(data_spain.Mean_Throughput) & (data_spain.Technology=="5G"))]["Mean_Throughput"]
    },
        'Vancouver': {
        'LTE': data_canada[(pd.notna(data_canada.Mean_Throughput) & (data_canada.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_canada[(pd.notna(data_canada.Mean_Throughput) & (data_canada.Technology=="5G"))]["Mean_Throughput"]
    },
    'Boston': {
        'LTE': data_usa_ma[(pd.notna(data_usa_ma.Mean_Throughput) & (data_usa_ma.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_usa_ma[(pd.notna(data_usa_ma.Mean_Throughput) & (data_usa_ma.Technology=="5G"))]["Mean_Throughput"]
    },
    'Bay Area': {
        'LTE': data_usa_ca[(pd.notna(data_usa_ca.Mean_Throughput) & (data_usa_ca.Technology=="LTE"))]["Mean_Throughput"],
        '5G': data_usa_ca[(pd.notna(data_usa_ca.Mean_Throughput) & (data_usa_ca.Technology=="5G"))]["Mean_Throughput"]
    }
}

# Create boxplots for each city
fig, ax = plt.subplots(figsize=(12, 6))
box_width = 0.4
colors = {'LTE': 'tab:blue', '5G': 'tab:orange'}
for i, city in enumerate(cities):
    positions = [i - box_width/2, i + box_width/2]
    for tech in ['LTE', '5G']:
        box_data = data_by_city[city][tech]
        ax.boxplot(box_data, positions=[positions[0] if tech == 'LTE' else positions[1]], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=colors[tech], color='black'), medianprops=dict(color='black'))
ax.set_ylim([-10, 175])        
ax.set_xticks(np.arange(len(cities)))
ax.set_xticklabels(cities)
# ax.set_xlabel('Cities')
ax.set_ylabel('Throughput [Mbps]')
# ax.set_title('Mean Latency of LTE and 5G in Different Cities')

# Create custom legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=colors[tech], label=tech) for tech in ['LTE', '5G']]
ax.legend(handles=handles)
plt.grid()  # Add gridlines



plt.savefig("Plots/Mean_Throughput_All.png", format="png", dpi=300)



import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
file_path = 'costtrain.csv'
time_path = 'timetrain.csv'
cost_df = pd.read_csv(file_path)
time_df = pd.read_csv(time_path)

# Replace '-' with NaN to mark missing connections, and set the 'City' column as index
cost_df.replace('-', np.nan, inplace=True)
cost_df.set_index('City', inplace=True)
time_df.replace('-', np.nan, inplace=True)
time_df.set_index('City', inplace=True)

# Convert data to numeric where possible
cost_df = cost_df.apply(pd.to_numeric, errors='coerce')
time_df = time_df.apply(pd.to_numeric, errors='coerce')

# Calculate layover connections
for city_from in cost_df.index:
    for city_to in cost_df.columns:
        # Skip if there's already a direct connection
        if not np.isnan(cost_df.loc[city_from, city_to]):
            continue

        # Lists to hold possible layover costs and times
        possible_layover_costs = []
        possible_layover_times = []
        for layover_city in cost_df.index:
            # Check for a valid layover route in both cost and time matrices
            if (layover_city != city_from and layover_city != city_to
                and not np.isnan(cost_df.loc[city_from, layover_city])
                and not np.isnan(cost_df.loc[layover_city, city_to])
                and not np.isnan(time_df.loc[city_from, layover_city])
                and not np.isnan(time_df.loc[layover_city, city_to])):

                # Calculate layover cost and time, converting to int
                layover_cost = int(cost_df.loc[city_from, layover_city] + cost_df.loc[layover_city, city_to])
                layover_time = int(time_df.loc[city_from, layover_city] + time_df.loc[layover_city, city_to])
                possible_layover_costs.append(layover_cost)
                possible_layover_times.append(layover_time)

        # If there are valid layover routes, pick the minimum cost and the corresponding time
        if possible_layover_costs and possible_layover_times:
            min_cost_index = possible_layover_costs.index(min(possible_layover_costs))
            cost_df.loc[city_from, city_to] = possible_layover_costs[min_cost_index]
            time_df.loc[city_from, city_to] = possible_layover_times[min_cost_index]


cost_df = cost_df.fillna(0).astype(int)
# Save the updated DataFrame back to a CSV file
cost_df.replace(0, '-', inplace=True)
output_path = 'updated_cost_matrix.csv'
cost_df.to_csv(output_path)

time_df = time_df.fillna(0).astype(int)
# Save the updated DataFrame back to a CSV file
time_df.replace(0, '-', inplace=True)
output_path = 'updated_time_matrix.csv'
time_df.to_csv(output_path)

print(f"Updated cost matrix with layovers saved to {output_path}")

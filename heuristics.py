import pandas as pd

def nested_dict_maker(file_path):
    data = pd.read_csv(file_path)

    nested_dict = {}

    for i, row in data.iterrows():
        origin_city = row['City']
        nested_dict[origin_city] = {destination: (int(time) if time != '-' else str(time)) for destination, time in row.items() if destination != 'City'}

    return nested_dict

def find_N_best_connections(conections_time, conections_price, cities, N):
    best_cities = {}
    counter = 0
    skipped = 0

    for city in cities:
        if conections_time[city] == "-" or conections_price[city] == "-":
            skipped += 1
            continue
        weighted_sum = 0.9 * conections_time[city] + 0.1 * conections_price[city]
        if counter < N:
            best_cities[city] = weighted_sum
            # if max_weighted_sum_of_5 < weighted_sum:
            #     max_weighted_sum_of_5 = weighted_sum
        else:
            if max(best_cities.values()) > weighted_sum:
                best_cities.pop(max(best_cities, key=best_cities.get))
                best_cities[city] = weighted_sum
        counter += 1

    return list(best_cities.keys())

def init_heuristics():
    best_connections = {}
    for city in dict_keys:
        best_connections[city] = find_N_best_connections(dict_time_plane[city], dict_cost_plane[city], dict_keys, 10)

        for city_found in best_connections[city]:
            if city_found not in repeated_connections:
                repeated_connections[city_found] = 0
            repeated_connections[city_found] += 1

        print(best_connections[city])
    return

time_plane_file = "./datasets/timeplane.csv"
cost_plane_file = "./datasets/costplane.csv"

dict_time_plane = nested_dict_maker(time_plane_file)
dict_cost_plane = nested_dict_maker(cost_plane_file)

dict_keys = list(dict_time_plane.keys())


repeated_connections = {}



print(50*"==")
for city in dict_keys:
    if city in repeated_connections:
        print(f"Rpeated {city}: {repeated_connections[city]}")
    else:
        print(f"Rpeated {city}: 0")
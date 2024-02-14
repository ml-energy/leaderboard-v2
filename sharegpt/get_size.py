import json

# Replace 'your_file_path.json' with the actual path to your JSON file
file_path = 'ShareGPT_V3_filtered.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)

print(len(data_list))

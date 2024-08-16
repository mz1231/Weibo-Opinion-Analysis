# Read the 50,000 IDs into a list
with open('datasets/50k_ids.txt', 'r') as file:
    ids_50k = file.read().splitlines()

# Create a mapping of ID to its index in the 50k list
id_position_map = {id_: index for index, id_ in enumerate(ids_50k)}

# Read the 7,546 IDs into a list
with open('yes_predictions.txt', 'r') as file:
    ids = file.read().splitlines()

# Sort the 7,546 IDs based on their order in the 50k IDs list
sorted_ids = sorted(ids, key=lambda id_: id_position_map.get(id_, float('inf')))

# Save the sorted IDs to a new file
with open('sorted_yes_predictions.txt', 'w') as file:
    for id_ in sorted_ids:
        file.write(id_ + '\n')

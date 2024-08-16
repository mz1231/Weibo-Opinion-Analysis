import csv

# Define the input and output file paths
input_csv_path = 'datasets/50k.csv'
output_txt_path = 'datasets/50k_ids.txt'

# Extract IDs from the CSV and write them to a text file
with open(input_csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open(output_txt_path, mode='w', newline='', encoding='utf-8') as txt_file:
        for row in csv_reader:
            id_ = row[0]  # Assuming ID is the first column
            txt_file.write(id_ + '\n')

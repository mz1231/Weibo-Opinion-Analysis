import pandas as pd

# Load the Excel file
excel_file_path = 'datasets/50k.xlsx'
df = pd.read_excel(excel_file_path)

# Select columns by their positions (A, B, and F are 1st, 2nd, and 6th columns respectively)
df_selected = df.iloc[:, [0, 1, 35]]

# Save as CSV file
csv_file_path = 'datasets/50k.csv'
df_selected.to_csv(csv_file_path, index=False)

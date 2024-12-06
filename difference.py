import pandas as pd
from itertools import combinations


def add_pairwise_differences(file_path, output_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # List of columns to calculate pairwise differences
        columns = ["R", "G", "B", "UV"]

        # Compute pairwise differences
        for col1, col2 in combinations(columns, 2):
            diff_col_name = f"{col1}_minus_{col2}"
            df[diff_col_name] = df[col1] - df[col2]

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_path, index=False)
        print(f"Updated CSV file saved as '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage
input_file = "dataset2.csv"  # Replace with your file path
output_file = "diffdata2.csv"
add_pairwise_differences(input_file, output_file)

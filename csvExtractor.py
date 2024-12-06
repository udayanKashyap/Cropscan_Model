import json
import csv


def extractData(fileName):
    with open(fileName, "r") as file:
        data = json.load(file)

    voltageStrings = list(data["voltageArr"].values())

    # print(voltageStrings)

    # Example array of strings
    # voltageStrings = [
    #     "1.1, 2.2, 3.3, ..., 50.0:255:128:64:0.5:Info1",
    #     "1.5, 2.5, 3.5, ..., 50.5:200:100:50:0.8:Info2",
    # ]

    # Output CSV file name
    output_file = "voltage_data.csv"

    # Prepare CSV headers
    voltage_headers = [f"v{i}" for i in range(1, 51)]
    other_headers = ["details", "R", "G", "B", "UV"]
    headers = other_headers + voltage_headers

    # Process the strings and extract data
    rows = []
    for voltage_string in voltageStrings:
        # Split into voltage values and additional data
        parts = voltage_string.split(":")
        voltage_values = parts[0].split(",")
        additional_data = parts[1:]  # Keep in original order: R, G, B, UV, details

        # Move details to the start
        row = [additional_data[-1]] + additional_data[:-1] + voltage_values
        rows.append(row)

    # Write data to the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write headers
        writer.writerows(rows)  # Write rows

    print(f"Data has been written to {output_file}")


if __name__ == "__main__":
    extractData("cropscanData.json")

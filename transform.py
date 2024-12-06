import pandas as pd
import pprint


def transform(input_file, output_file):
    data = pd.read_csv(input_file)
    new_data = []
    classes = data["details"].unique()
    print(classes)
    for j in range(2, data.shape[1]):
        # for j in range(2, 4):
        # print(data.iloc[:, j].values)
        all_class_voltages = data.iloc[:, j].values
        for i in range(0, len(all_class_voltages), 4):
            new_row = {
                "details": classes[(i * 3) // len(classes)],
                "class": (i * 3) // len(classes),
                "R": all_class_voltages[i],
                "G": all_class_voltages[i + 1],
                "B": all_class_voltages[i + 2],
                "UV": all_class_voltages[i + 3],
            }
            new_data.append(new_row)
    transformed_data = pd.DataFrame(
        new_data, columns=["details", "class", "R", "G", "B", "UV"]
    )
    transformed_data.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = "./dataset/50data_raw.csv"
    output_file = "50data.csv"
    transform(input_file, output_file)

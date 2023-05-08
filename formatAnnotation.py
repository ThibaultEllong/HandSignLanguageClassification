from sklearn import preprocessing
import pandas as pd

# Converts the classes in the annotation file to categorical values

    
"""df = pd.read_csv('HandSignLanguageClassification/Dataset/Annotation_gloses.csv', sep='\t')

cols = list(df.columns)
action_class, label = cols[0], cols[1]

names = df[action_class].values
labels = df[label].values"""

def to_categorical(annotation_file):
    data = pd.read_csv(annotation_file, sep="\t")
    cols = list(data.columns)
    data.head()
    action_class, label = cols[0], cols[1]
    data[label] = data[label].apply(lambda x: 0 if x == 'Mono' else 1)
    data.to_csv(annotation_file, index=False)
    



def to_txt(csv_file, column_number):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Check if the given column_number is valid
    if column_number < 0 or column_number >= df.shape[1]:
        raise ValueError("Invalid column number")

    # Extract the required column using the column_number
    column_data = df.iloc[:, column_number]

    # Create a txt file with the same name as the CSV file
    txt_file = csv_file.replace(".csv", f"_column{column_number}.txt")

    # Write the column data to the txt file
    with open(txt_file, 'w') as file:
        for value in column_data:
            file.write(str(value) + "\n")

    print(f"Column {column_number} from {csv_file} has been saved in {txt_file}")

to_txt('Dataset/Annotation_Categorical.csv', 0)
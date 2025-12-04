import numpy as np
import pandas as pd

def read_excel_data(path):

    data = pd.read_excel(path)

    unique_values = set()

    # Iterate through the 'eko bulguları' column and collect unique values
    for i in list(data['eko bulguları']):
        if type(i) == float: continue
        for j in i.split(','):
            unique_values.add(j.strip())

    # Create a mapping of unique values to indices
    # The mapping will be a dictionary where the key is the unique value and the value is its index
    mapping = {v:idx for idx, v in enumerate(list(unique_values))}

    data['NT proBNP'] = (data['NT proBNP'] - data['NT proBNP'].mean()) / data['NT proBNP'].std()

    # Initialize a dictionary to hold the train, validation, and test splits
    train_val_test_features = {"train": {}, "val": {}, "test": {}}

    # Iterate through the DataFrame rows and populate the train_val_test dictionary
    for _, row in data.iterrows():
        idx = []

        if type(row['eko bulguları']) != float:
            # Split the string by commas and map each value to its corresponding index
            for i in row['eko bulguları'].split(','):
                if type(i) == float: continue
                idx.append(mapping[i.strip()])

        # Create a zero-initialized array of length equal to the number of unique values + 4 for the additional features
        # and set the corresponding indices to 1
        record = np.zeros(len(mapping) + 4)
        record[idx] = 1
    
        record[-4:] = [row['kreatinin'], row['NT proBNP'] if row['NT proBNP'] != 'yok' else 0, row['eko EF'], row['3. SAAT K/KL']]
        
        train_val_test_features[row['Split']][row['HASTA ADI']] = record.tolist()

    return train_val_test_features
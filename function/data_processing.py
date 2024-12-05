# data_processing.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def load_and_process_data(file_path, num_zero_threshold):
    df = pd.read_csv(file_path)
    molecules = df.Smiles.apply(Chem.MolFromSmiles)
    fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())
    
    HashCode = []
    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)
    
    unique_set = set(HashCode)
    unique_list = list(unique_set)
    Corr_df = pd.DataFrame(unique_list).reset_index()
    
    MY_finger = []
    for polymer in fp_n:
        my_finger = [0] * len(unique_list)
        for key in polymer.keys():
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]
        MY_finger.append(my_finger)
    
    X = pd.DataFrame(MY_finger)
    Zero_Sum = (X == 0).astype(int).sum()
    Columns = Zero_Sum[Zero_Sum < num_zero_threshold].index
    X_count = X[Columns]
    Y = df['Tm'].values
    
    return X_count, Y
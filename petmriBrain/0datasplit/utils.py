import pandas as pd
import pickle


def save_csv(csv_rows,csv_path,name_attribute):
    with open(csv_path, mode='w') as file:
        writerCSV = pd.DataFrame(columns=name_attribute, data=csv_rows)
        writerCSV.to_csv(csv_path, encoding='utf-8', index=False)
        
def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def save_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)
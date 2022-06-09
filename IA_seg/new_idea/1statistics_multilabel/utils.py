import pandas as pd
import pickle
import SimpleITK as sitk
import os

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
        
        
def maybe_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_csv(csv_rows,csv_path,name_attribute):
    with open(csv_path, mode='w') as file:
        writerCSV = pd.DataFrame(columns=name_attribute, data=csv_rows)
        writerCSV.to_csv(csv_path, encoding='utf-8', index=False)
    
def save_itk_from_numpy(numpy_data, property):

    pred_itk_image = sitk.GetImageFromArray(numpy_data)
    pred_itk_image.SetSpacing(property["itk_spacing"])
    pred_itk_image.SetOrigin(property["itk_origin"])

    return pred_itk_image




def generate_datainfo_csv(name,small_n,middle_n,large_n,d,v,properti,datainfo):
 
    #name_attribute = ['instance_id','institution_id','age','gender','is_ruptured','num_IAs','spacing','slices','size','small_IAs_num','middle_IAs_num','large_IAs_num','diameters','voxels']
    
    img = dict()
    assert datainfo['instance_id'] == name
    assert properti['id'] == name
    img['id'] = name
    
    img['institution_id'] = datainfo['institution_id']
    img['age'] = datainfo['age']
    img['gender'] = datainfo['gender']
    img['is_ruptured'] = datainfo['is_ruptured']
    img['num_IAs'] = datainfo['num_IAs']
    
    img['spacing'] = properti['itk_spacing']
    img['slices'] = properti['slices']
    img['size'] = properti['after_size']
    
    img['small_IAs_num'] = small_n
    img['middle_IAs_num'] = middle_n
    img['large_IAs_num'] = large_n
    
    img['diameters'] = d
    img['voxels'] = v
    
    return img


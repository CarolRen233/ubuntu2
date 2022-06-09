import json
import numpy as np
import argparse
import os
import pandas as pd
'''
parser = argparse.ArgumentParser(description='AneurysmSeg study evaluation')
parser.add_argument('-p', '--pred_path', type=str, required=True, default='',
                    help='pred path')

args = parser.parse_args()
'''

pred_path = '/root/workspace/reny/petmriBrain/code/3nnUNet-master/output/nnUNet_trained_models/nnUNet/3d_fullres/Task200_mriBrain/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/Task200_best_infer'



def save_csv(csv_rows,csv_path,name_attribute):
    with open(csv_path, mode='w') as file:
        writerCSV = pd.DataFrame(columns=name_attribute, data=csv_rows)
        writerCSV.to_csv(csv_path, encoding='utf-8', index=False)
        
def generate_singlecase_csv(flag,summary_info):
 
    #flag = ['Accuracy','Dice','False Negative Rate','False Positive Rate','Jaccard',
    #'Precision','Recall','True Negative Rate']
    
    print ('========single case=============')
    
    csv_list = []
    for singlecase in summary_info:
        img = dict()
        print ('singlecase[reference]:',singlecase['reference'])
        print ('name:',singlecase['reference'].split('/')[-1][:-7])
        img['name'] = singlecase['reference'].split('/')[-1][:-7]
        for i in range(1,45):
            img[str(i)] = singlecase[str(i)][flag]
        csv_list.append(img)
    
    return csv_list


def generate_label_csv(summary_info):
    
    print ('========label=============')
 
    flags = ['Accuracy','Dice','False Negative Rate','False Positive Rate','Jaccard','Precision','Recall','True Negative Rate']
    
    csv_list = []
    for i,val in summary_info.items():
        img = dict()
        img['label'] = str(i)
        for fg in flags:
            print ('flag:',fg)
            img[fg] = val[fg]
        csv_list.append(img)
    return csv_list


if __name__ == '__main__':
    # pred_path = args.pred_path
    
    with open(os.path.join(pred_path, 'summary.json')) as f:
        summary = json.load(f)
    
    singlecase_name_attribute = ['name']
    for i in range(1,45):
        singlecase_name_attribute.append(str(i))
    
    flags = ['Accuracy','Dice','False Negative Rate','False Positive Rate','Jaccard','Precision','Recall','True Negative Rate']
    for flag_name in flags:
        evaluate_list = generate_singlecase_csv(flag_name,summary['results']['all'])
        save_csv(evaluate_list,os.path.join(pred_path,flag_name+'.csv'),singlecase_name_attribute)
        print (flag_name+'.csv saved!')
        
    label_list = generate_label_csv(summary['results']['mean'])
    save_csv(label_list,os.path.join(pred_path,'labels_evaluate.csv'),['label']+flags)
    print ('labels.csv saved!')
    
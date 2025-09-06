import os
import csv
import pandas as pd
from pathlib import Path

source = "csv_16_batch/"
target = "summary/"

header = ['Fold', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']

header_summary = [' ','Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
#'Average Training Loss', 'Average Training Accuracy', 'Average Validation Loss', 'Average Validation Acc']

experiments = [d for d in os.listdir(source) if os.path.isdir(source+d)]





for experiment in experiments:
    path = source + experiment
    df_data = pd.DataFrame(columns=header)
    df_summary = pd.DataFrame(columns=header_summary)
    for filename in os.listdir(path+"/train/acc"):
        df_train_acc = pd.read_csv(path+"/train/acc/"+filename)
        df_train_loss = pd.read_csv(path+"/train/loss/"+filename)
        df_val_acc = pd.read_csv(path+"/val/acc/"+filename)
        df_val_loss = pd.read_csv(path+"/val/loss/"+filename)
        
        #print(filename)
        new_row = {'Fold': filename.split(".")[0], 'Training Loss': round(df_train_loss['Value'].mean(),4), 'Training Accuracy': round(df_train_acc['Value'].mean(),4), 'Validation Loss': round(df_val_loss['Value'].mean(),4), 'Validation Accuracy': round(df_val_acc['Value'].mean(),4)}
        df_data.loc[len(df_data)] = new_row
    
    df_summary.loc[len(df_summary)] = {' ': 'Mean', 'Training Loss': round(df_data['Training Loss'].mean(),4), 'Training Accuracy': round(df_data['Training Accuracy'].mean(),4), 'Validation Loss': round(df_data['Validation Loss'].mean(),4), 'Validation Accuracy': round(df_data['Validation Accuracy'].mean(),4)}
    df_summary.loc[len(df_summary)] = {' ': 'Std', 'Training Loss': round(df_data['Training Loss'].std(),4), 'Training Accuracy': round(df_data['Training Accuracy'].std(),4), 'Validation Loss': round(df_data['Validation Loss'].std(),4), 'Validation Accuracy': round(df_data['Validation Accuracy'].std(),4)}
    df_summary.loc[len(df_summary)] = {' ': 'Min', 'Training Loss': round(df_data['Training Loss'].min(),4), 'Training Accuracy': round(df_data['Training Accuracy'].min(),4), 'Validation Loss': round(df_data['Validation Loss'].min(),4), 'Validation Accuracy': round(df_data['Validation Accuracy'].min(),4)}
    df_summary.loc[len(df_summary)] = {' ': 'Max', 'Training Loss': round(df_data['Training Loss'].max(),4), 'Training Accuracy': round(df_data['Training Accuracy'].max(),4), 'Validation Loss': round(df_data['Validation Loss'].max(),4), 'Validation Accuracy': round(df_data['Validation Accuracy'].max(),4)}
    df_summary.loc[len(df_summary)] = {' ': 'Median', 'Training Loss': round(df_data['Training Loss'].median(),4), 'Training Accuracy': round(df_data['Training Accuracy'].median(),4), 'Validation Loss': round(df_data['Validation Loss'].median(),4), 'Validation Accuracy': round(df_data['Validation Accuracy'].median(),4)}

    #print(df_data.to_string())
    #print(df_summary.to_string())
    pathname = target+experiment + "/" 
    path = Path(pathname)
    path.mkdir(parents=True, exist_ok=True)
    df_data.sort_values(by=['Fold'], inplace=True)
    df_data.to_csv(target+experiment+"/"+experiment+".csv", index=False)
    df_summary.to_csv(target+experiment+"/"+experiment+"_summary.csv", index=False)



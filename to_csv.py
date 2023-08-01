import os
import numpy as np
import pandas as pd
import torch
import csv

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_data(path, type='train/loss'):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    for e in event_acc.Scalars(type):
        print(e.step, e.value)


def write_csv(dir,fold,headers,data):
    print("Writing fold",fold,"to csv in: ",dir)
    with open(dir+"fold"+str(fold)+".csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    steps_per_epoch=18
    experiments=["MobileNetV3Small_exp21_1000epoch_10fold_3segment_1frampersegment_batchsize16_optSGD"]
    paths = ["runs/MobileNetV3Small_exp21_1000epoch_10fold_3segment_1frampersegment_batchsize16_optSGD/events.out.tfevents.1689794694.carthago.408883.0"]
    #print_data(paths)
    fields = ['Step', 'Value']
    graph_types = ['train/loss', 'train/acc', 'val/loss', 'val/acc']

    
    for i in range(len(paths)):
        event_acc = EventAccumulator(paths[i])
        print(event_acc.Reload())
        print(event_acc.Tags())
        for graph_type in graph_types:
            dir = "csv/"+ experiments[i] +"/"+ graph_type + "/"
            os.makedirs(dir,exist_ok=True)
            #train_loss = create_csv(paths[i],dir,fields,graph_type,steps_per_epoch=9)
            flag = False
            train_loss = list()
            fold = -1
            if graph_type == 'val/loss' or graph_type == 'val/acc':
                #print(type)
                flag = True
            
            for e in event_acc.Scalars(graph_type):
                step = e.step
                value = e.value
                # if flag:
                #     print(e.step, e.value)
                if step == 0 or (flag == True and step == (steps_per_epoch-1)):
                    fold += 1
                    if fold != 0:
                        write_csv(dir,fold,fields,train_loss)
                        train_loss.clear()
        
                train_loss.append({fields[0]: step, fields[1]: value})
            write_csv(dir,fold+1,fields,train_loss)
    
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
source = "csv_16_batch/"
target = "plots_16_batch/"

steps_per_epoch = 18
### create target directory if it doesn't exist ###
os.makedirs(target,exist_ok=True)

experiments = [d for d in os.listdir(source) if os.path.isdir(source+d)]
# for experiment in experiments:
#     print(experiment)

# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

#print("Current Working Directory ", os.getcwd())

# Read a CSV file

def Average(lst):
    return sum(lst) / len(lst)

########## TODO:  ##########
#experiments = [experiments[1]]
############################
for experiment in experiments:
    print(experiment)
    temp = os.listdir(source+experiment+"/train/acc/")
    count = len(temp)
    #print(temp, len(temp))
    
    for i in range(count): # for each fold
        # Make a list of columns
        for metric in ['acc','loss']: # for each metric
            columns = ['Step', 'Value']
            legends = ['train', 'val']
            #print(legends)
            df_train = pd.read_csv(source + experiment + "/train/" +metric +"/fold"+str(i+1)+".csv", usecols=columns)
            #df_train.set_index('Step', inplace=True)
            df_val = pd.read_csv(source + experiment + "/val/" +metric + "/fold"+str(i+1)+".csv", usecols=columns)
            #df_val.set_index('Step', inplace=True)
            #print(df_train==df_val)
            df_train_mean = pd.DataFrame(columns=['Epoch', 'Value'])
            df_val_mean = pd.DataFrame(columns=['Epoch', 'Value'])
            

            #counter = 0
            epoch_id = 0
            train_values = list()
            val_values = list()
            for _,row in df_train.iterrows():
                #print(row)
                step =row['Step']
                value = row['Value']
                
                index_start = epoch_id * steps_per_epoch
                index_end = index_start + steps_per_epoch
                
                if step >= index_start and step <= index_end:
                    train_values.append(value)
                else:
                    if len(train_values) != 0:
                        df_train_mean.loc[epoch_id] = [epoch_id, Average(train_values)]
                    train_values = list()
                    train_values.append(value)
                    epoch_id += 1
                #counter = counter + 1

            #counter = 0
            epoch_id = 0
            for _,row in df_val.iterrows():
                step =row['Step']
                value = row['Value']

                index_start = epoch_id * steps_per_epoch
                index_end = index_start + steps_per_epoch

                if step >= index_start and step <= index_end:
                    val_values.append(value)
                else:
                    if len(val_values) != 0:
                        df_val_mean.loc[epoch_id] = [epoch_id, Average(val_values)]
                    val_values = list()
                    val_values.append(value)
                    epoch_id += 1 
                #counter = counter + 1

                #train_values = [value for value in df_train[index_start:index_end]['Value']]
                #print(train_values)
            #     df_train_mean.loc[epoch_id] = [epoch_id, Average(train_values)]
            #     val_values = [value for value in df_val[index_start:index_end]['Value']]
            #     print(val_values)
            #     df_val_mean.loc[epoch_id] = [epoch_id, Average(val_values)]
            #     counter = index_end
            #     epoch_id += 1
            # print(df_val_mean)
    
            plt.title(experiment + " " + metric + " fold " + str(i))
            plt.ylim(0, 1)
            #plt.set_ylim(0, 1)
            #plt.xlabel('Epoch')
            #plt.ylabel(metric)
            ax1 = df_train_mean.plot(x='Epoch', y='Value', kind = 'line') #legend=True, label=legends)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel(metric)
            ax2 = df_val_mean.plot(ax = ax1,x='Epoch', y='Value', kind = 'line')#,legend=True, label=legends)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric)
            pathname = target+experiment + "/"+ metric + "_train-val/" 
            path = Path(pathname)
            path.mkdir(parents=True, exist_ok=True)
            plt.legend(legends)
            plt.savefig(pathname+"fold"+str(i+1)+".png")
            plt.clf()
            plt.close()


    
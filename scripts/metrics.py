import os
from striprtf.striprtf import rtf_to_text

result_dir = './results'
labels_dir = './GT/TEST_SET/ALL'

PFR_TARGET = 10
MEM_TARGET = 4
TOTAL_FRAMES = 1065
TOTAL_TIME =  0.15652179718017578

####### LOADING PREDICTIONS #######
predictions = {}
predictions_filenames = [f for f in os.listdir(result_dir) if f.endswith(".txt")]

for filename in predictions_filenames:
    
    if os.stat(result_dir+"/"+filename).st_size == 0:
        filename = filename.split(".")[0]
        predictions[filename] = -1
    else:
        with open(result_dir+"/"+filename, "r") as file:
            f = file.readlines()
        filename = filename.split(".")[0]
        predictions[filename] = int(f[0])


####### LOADING LABELS #######
labels = {}
labels_filenames = [f for f in os.listdir(labels_dir) if f.endswith(".rtf")]

for filename in labels_filenames:
    with open(labels_dir+'/'+filename, 'r') as file:
        text = rtf_to_text(file.read())
    #print(filename," ",text)
    filename = filename.split(".")[0]
    if len(text): # if text is not empty
        labels[filename] = text.split(",")[0]
    else:
        labels[filename] = -1
        


####### PRINT #######
print('\n\nPREDICTIONS\n')
for k,g in predictions.items():
    print(k + ' ' + str(g))

print('\n\nLABELS\n')
for k,g in labels.items():
  print(k + ' ' + str(g))


####### METRICHE #######

if(len(predictions) != len(labels)):
  print('I due set hanno dimensioni diverse:')
  print('labels: ' + str(len(labels)))
  print('predictions: ' + str(len(predictions)))
  print('I video mancanti sono: ')
  for k,g in labels.items():
    try:
      elem = predictions[k]
    except:
      print(k)

delta_t = 5
TN = TP = FP = FN = 0

true_positive_list = []
false_positive_list = []
true_negative_list = []
false_negative_list = []

####### TRUE POSITIVE #######
for k,g in labels.items():
  g=int(g)
  if(g != -1):
    p = int(predictions[k])
    if p != -1 and p >= max(0,g-delta_t):   # predire in ritardo NON è un problema
      TP = TP + 1
      true_positive_list.append(k)

####### FALSE POSITIVE #######
for k,g in labels.items():
  g=int(g)
  p = int(predictions[k])
  if g == -1:
    if p != -1:
      FP = FP + 1
      false_positive_list.append(k)
  elif p != -1 and p < max(0,g-delta_t):  # predire un tempo antecedente al reale tempo di comparsa del fuoco porta ad avere un falso positivo
    FP = FP + 1
    false_positive_list.append(k)

####### FALSE NEGATIVE #######
for k,g in labels.items():
  g=int(g)
  if(g != -1):
    p = int(predictions[k])
    if p == -1:
      FN = FN + 1
      false_negative_list.append(k)



try:
  P = abs(TP)/(abs(TP)+abs(FP))
except:
  P = 0

try:
  R = abs(TP)/(abs(TP)+abs(FN))
except:
  R = 0

####### TRUE NEGATIVE #######
for k,g in labels.items():
  g=int(g)
  if(g == -1):
    p = int(predictions[k])
    if p == -1:
      TN = TN + 1
      true_negative_list.append(k)

TN = len(labels)-TP-FP-FN

accuracy = (TP+TN)/len(labels)

####### DELAY #######
d = {}
for k in true_positive_list:
  g = int(labels[k])
  p = int(predictions[k])
  d[k] = abs(p-g)

if TP == 0:
  print('TP = 0, impossible to calculate Delay, because there are no true positives')
  D = 0
  Dn = 0
else:
  D = sum(d.values())/TP
  Dn = max(0,60-D)/60


####### RESULTS #######
print('TP: ' + str(TP))
print('TN: ' + str(TN))
print('FP: ' + str(FP))
print('FN: ' + str(FN))
print("true positive list: ",true_positive_list)
print("true negative list: ",true_negative_list)
print("false positive list: ",false_positive_list)
print("false negative list: ",false_negative_list)
print("\n\n")

print('accuracy: ' + str(round(accuracy, 3)))
print('precision: ' + str(round(P, 3)))
print('recall: ' + str(round(R, 3)))

print("R*P: ",round(R*P,3))

print('average delay: ' + str(round(D, 3)))
print('normalized average delay: ' + str(round(Dn, 3)))

PFR = 1/(TOTAL_TIME/TOTAL_FRAMES)

PFR_delta = max(0,PFR_TARGET/PFR - 1)
print('PFR:',str(round(PFR,3)),"PFR_delta: " + str(round(PFR_delta, 3)))

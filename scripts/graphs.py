import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

LOSS_PATH="FireNetV2_400epoch_10fold_3segment_1frampersegment_batchsize32/ignore/val_losses.pth"
ACCURACY_PATH="FireNetV2_400epoch_10fold_3segment_1frampersegment_batchsize32/ignore/val_accuracies.pth"

### validation losses and accuracies graphs ###
val_losses=torch.load(LOSS_PATH)
val_accuracies=torch.load(ACCURACY_PATH)
K_cross_val=5
epoch=400

fold_val_losses=torch.transpose(val_losses,0,1)
# for i in range(0,epoch):
#   if(fold_val_losses[0][i]!=val_losses[i][0]):
#     print(fold_val_losses[0][i])

for i in range(K_cross_val):
  x=range(0,epoch)
  plt.plot(x,fold_val_losses[i])
  plt.legend(["Fold " + str(i)])
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.show()
  #plt.savefig('/content/gdrive/MyDrive/ML/ResNet50_exp1_200epoch_10fold_3segment_1framepersegment_32batchsize/Val_losses/val_loss_fold_{}.png'.format(i))

fold_val_acc=torch.transpose(val_accuracies,0,1)
for i in range(0,200):
  if(fold_val_acc[0][i]!=fold_val_acc[i][0]):
    print(fold_val_acc[0][i])

for i in range(K_cross_val):
  x=range(0,epoch)
  plt.plot(x,fold_val_acc[i])
  plt.legend(["Fold " + str(i)])
  plt.xlabel("epoch")
  plt.ylabel("accuracy")
  plt.show()

### 
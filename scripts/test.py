import cv2, os, argparse
import torch
import albumentations
from torchvision import transforms
from models import *
#import time
from firenet import *
from firenetV2 import FireNetV2

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args
args = init_parameter()


# Here you should initialize your method
WEIGHT_PATH = 'ResNet50_exp13_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/ResNet50_exp13_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/fold_9_best_model.pth'
MIN_DURATION = 10
THRESHOLD = 0.5

#####  MODEL CREATION #####

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_ResNet50(num_outputs=1)
model.load_state_dict(torch.load(WEIGHT_PATH,map_location=device))

model = model.cuda() if torch.cuda.is_available() else model
model.eval()

#print('Videos folder: {}'.format(args.videos),"Current working directory: ", os.getcwd())
num_videos: int = len(os.listdir(args.videos))
#print('Numero di video: {}'.format(num_videos))

total_frames: int = 0
total_time = 0
num_frames: int = 0
num_imgs: int = 0
start_frame_fire = None
fire_duration: int = 0
video_prediction = None
################################################

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        
        #start_time = time.time()
        if ret == True:
            if num_frames % fps == 0: # Prendo un frame ogni secondo
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ##### PREPROCESSING IMAGES #####

                transform = albumentations.Sequential([
                    albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True),
                    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 max_pixel_value=255.,
                                 always_apply=True),
                ])
                
                img = transform(image=img)["image"]
                
                with torch.no_grad():
                    
                    input_batch = transforms.functional.to_tensor(img).unsqueeze(0) # create a mini-batch as expected by the model        
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        
                    output = model(input_batch)
                
                ###################### 
                prediction = torch.nn.functional.sigmoid(output[0]) 
                ######################
               
                ####### CLASSIFICATION LOGIC #######
                if prediction >= THRESHOLD: # Fire detected in the current frame
                    #print("prediction >= THRESHOLD: ", num_imgs)
                    if start_frame_fire is None:
                        start_frame_fire = num_imgs
                    fire_duration += 1
                    #print("prediction >= THRESHOLD. fire_duration: ",fire_duration, "start_frame_fire: ", start_frame_fire)
                else: # No fire detected in the current frame
                    #print("prediction < THRESHOLD. prediction: ",prediction)
                    fire_duration = 0
                    start_frame_fire = None

                if fire_duration >= MIN_DURATION: # Fire detected for at least MIN_DURATION seconds
                    #print("Fire detected for at least MIN_DURATION seconds. start_frame_fire: ",start_frame_fire,"num_imgs" ,num_imgs)
                    img = None
                    num_imgs += 1
                    video_prediction = 1
                    break
                        
                num_imgs += 1
            num_frames += 1
            img = None

    #end_time = time.time()

        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")

    # Here you should add your code for writing the results
    if video_prediction is None and start_frame_fire == 0 and prediction >= THRESHOLD: 
        # If the fire video is shorter than MIN_DURATION, 
        # we consider it as a fire video if: the first frame is classified as fire (start_frame_fire == 0 ) and the last frame is also classified as fire. 
        # This means that all frames of the video are classified as fire.
        video_prediction = 1

    if video_prediction:
        t = int(start_frame_fire)
        f.write(str(t))

    total_frames += num_imgs
    #total_time += end_time-start_time
    #end_time = 0
    #start_time = 0

    num_frames = 0
    num_imgs = 0
    start_frame_fire = None
    fire_duration = 0
    prediction = None
    video_prediction = None
    ########################################################
    f.close()

#print("Total frames: ", total_frames)
#print("Total time: ", total_time)




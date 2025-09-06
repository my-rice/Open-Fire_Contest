import os
source = "../../Prova/TestSet/Smoke"
target = "./mp4/"

# This script converts all .avi files in source directory to .mp4 using ffmpeg and store them in target directory

print("Current Working Directory ", os.getcwd())
os.chdir(source)

filenames =  [f for f in os.listdir('.') if f.endswith(".avi")]

for f in os.listdir('.'):
    print(f)

for filename in filenames:
    output_name = target + filename[:-4] + ".mp4"
    print(filename,output_name)
    #os.system("ffmpeg -i {} -c:v copy -c:a copy {}".format(filename, output_name))
    #os.system("ffmpeg -i {} -codec copy {}".format(filename, output_name))
    os.system("ffmpeg -i {} -strict -2 {}".format(filename, output_name))
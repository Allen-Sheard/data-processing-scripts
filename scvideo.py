"""随机截取视频片段保存"""

import cv2
import random
import os
import glob

def writevideo(inputvideopath,outputvideopath,scale,isdao=True):
    cap = cv2.VideoCapture(inputvideopath)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    frame_num = cap.get(7)
    out = cv2.VideoWriter(outputvideopath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))
    frameIndex = frame_num - 1
    frame_write_num = frame_num*scale
    for i in range(int(frame_write_num)):
        if(isdao):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)  # 视频写入
            frameIndex -= 1
        else:
            break
    cap.release()
    out.release()
data_path = "E:/datasets/yjsk_test/yjsk_videos1"
save_path = "E:/datasets/yjsk_test/yjsk_videos3/"
video_path_list = glob.glob(data_path+"/*.mp4")

for video_path in video_path_list[:25]:
    video_base_name = os.path.basename(video_path)[:4]
    numbervideo = 30
    for id in range(numbervideo):
        outputvideoname = video_base_name+"_"+str(id)+".mp4"
        outputvideopath = os.path.join(save_path,outputvideoname)
        frame_scale = random.uniform(0.2, 0.8)
        isdao = bool(random.getrandbits(1))
        writevideo(video_path, outputvideopath, frame_scale, isdao=isdao)

"""
按各类包含的标签分类,保存在新文件夹
数据集分成person,bj,hxq三类

"""

import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import multiprocessing

# 三大类包含的各自标签名称
person = ["wcaqm","wcgz","xy","aqmzc","gzzc"]
bj = ["bj_bpmh","bj_bpps","bj_wkps","bjdsyc","bjdszc","bj_bpzc"]
hxq = ["hxq_gjtps","hxq_gjbs","hxq_gjzc","ywzt_yfyc"]
with open('./vscode-project/python/classes.txt','r') as f:
    classes = f.read().strip().split()

# 读取数据原始xml标签文件，统计各类数据量
def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    person_num,bj_num,hxq_num=0,0,0
    for obj in root.iter('object'):
        label = obj.find('name').text
        if label=='person':
            person_num+=1
        elif label=='bj':
            bj_num+=1
        else:
            hxq_num+=1
    print("{}person,{}bj,{}hxq".format(person_num,bj_num,hxq_num))
    return person_num,bj_num,hxq_num

# 从原数据集中分离各类数据，统计各类图片总量及框总数
def split(image_list,label_src,image_dst):
    person_path = os.path.join(image_dst,'person\\')
    if os.path.exists(person_path) is False:
        os.mkdir(person_path)
    bj_path = os.path.join(image_dst,'bj\\')
    if os.path.exists(bj_path) is False:
        os.mkdir(bj_path)
    hxq_path = os.path.join(image_dst,'hxq\\')
    if os.path.exists(hxq_path) is False:
        os.mkdir(hxq_path)
    numimage=1
    person_box,bj_box,hxq_box=0,0,0
    person_img,bj_img,hxq_img = 0,0,0
    for image_path in image_list:
        image_base_name = os.path.basename(image_path)
        label_name = image_base_name.split('.jpg')[0]+'.xml'
        label_path = os.path.join(label_src,label_name)
        if os.path.exists(label_path) is False:
            continue
        print("{}/{} now read {}".format(str(numimage),str(len(image_list)),label_name))
        p_num,b_num,h_num = read_xml(label_path)
        person_box+=p_num
        bj_box+=b_num
        hxq_box+=h_num
        if p_num>0:
            shutil.move(image_path, os.path.join(person_path,image_base_name))
            person_img += 1
        elif b_num>0:
            shutil.move(image_path, os.path.join(bj_path,image_base_name))
            bj_img += 1
        elif h_num>0:
            shutil.move(image_path, os.path.join(hxq_path,image_base_name))
            hxq_img += 1
        numimage+=1
    print('一共有{}张person图片,{}张bj图片,{}张hxq图片'.format(person_img,bj_img,hxq_img))
    print('{} person boxes,{} bj boxes,{} hxq boxes'.format(person_box,bj_box,hxq_box))

if __name__=="__main__":
    image_src = '.\\datasets\\JPEGimages\\'				# 原始数据图片目录
    image_dst = '.\\datasets\\1\\'						# 保存图片目录
    label_dst = ".\\datasets\\xml\\"					# 保存标签目录
    if os.path.exists(image_dst) is False:
            os.mkdir(image_dst)
	if os.path.exists(label_dst) is False:
			os.mkdir(label_dst)
    works = 2											# 进程数
    new_image_list = []
    image_list = glob.glob(image_src+"*.jpg")[:30000]	#只提取前30000张
    oneworkvideonum=int(len(image_list)/works)
    for i in range(works-1):
        new_image_list.append(image_list[oneworkvideonum*i:oneworkvideonum*(i+1)])
    new_image_list.append(image_list[oneworkvideonum*(works-1):])
    m_lsit=[]
    workid=0
    for j in range(works):
        m1=multiprocessing.Process(target=split,kwargs={"image_list":new_image_list[workid],"label_src":label_dst,"image_dst":image_dst})
        workid+=1
        m_lsit.append(m1)
    for m in m_lsit:
        m.start() 
    for m in m_lsit:
        m.join()
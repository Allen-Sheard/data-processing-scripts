"""
 将数据集中部分标签合并成一个大类保存为新的标签文件
"""
import time
import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import multiprocessing

person = ["wcaqm","wcgz","xy","aqmzc","gzzc"]
bj = ["bj_bpmh","bj_bpps","bj_wkps","bjdsyc","bjdszc","bj_bpzc"]
hxq = ["hxq_gjtps","hxq_gjbs","hxq_gjzc","ywzt_yfyc"]

with open('./vscode-project/python/classes.txt','r') as f:
    classes = f.read().strip().split()
# print(classes)

# 分离各类标签，移动图片复制标签，保存到新数据集目录下
def move(image_list,label_src,image_dst):
    # image_list = glob.glob(image_src+'*.jpg')
    # print('一共有%d张图'%len(image_list))
    for image_path in image_list:
        image_base_name = os.path.basename(image_path)

        label_name = image_base_name.split('.jpg')[0]+'.txt'
        label_path = os.path.join(label_src,label_name)
        if os.path.exists(label_path) is False:
            continue
        print('now read ',label_name)
        person_img,bj_img,hxq_img = 0,0,0
        with open(label_path, 'r') as f:
            person_num,bj_num,hxq_num = 0,0,0
            for i in f.readlines():
                index = i.split(' ')[0] 
                if classes[int(index)] in person:
                    shutil.move(image_path, os.path.join(os.path.join(image_dst,'person\\'),image_base_name))
                    shutil.copy(label_path, os.path.join(label_dst,label_name))
                    person_num += 1
                    continue
                elif classes[int(index)] in bj:
                    shutil.move(image_path, os.path.join(os.path.join(image_dst,'bj\\'),image_base_name))
                    shutil.copy(label_path, os.path.join(label_dst,label_name))
                    bj_num += 1
                    continue
                elif classes[int(index)] in hxq:
                    shutil.move(image_path, os.path.join(os.path.join(image_dst,'hxq\\'),image_base_name))
                    shutil.copy(label_path, os.path.join(label_dst,label_name))
                    hxq_num += 1
                    continue
            if person_num!=0:
                person_img+=1
            elif bj_num!=0:
                bj_img+=1
            elif hxq_num!=0:
                hxq_img+=1
            print('move down,person:{},bj:{},hxq:{}'.format(person_num,bj_num,hxq_num))
    print('一共有{}张person图片,{}张bj图片,{}张hxq图片'.format(person_img,bj_img,hxq_img))


def Iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1) 
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    min_rect = [min(xmin1,xmin2), min(ymin1,ymin2), max(xmax1,xmax2), max(ymax1,ymax2)]	#最小外接矩形

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  
    a2 = s1 + s2 - a1
    iou = a1 / a2           # 两框交并比
    area_ratio = a1 / s1    # 两框交集和面积较小的之比
    return iou, area_ratio, min_rect

# yolo格式转voc格式
def yolo2voc(box,img_w,img_h):
    x_center = float(box[0])*img_w+1
    y_center = float(box[1])*img_h+1
    xmin = int(x_center - 0.5*float(box[2])*img_w)
    ymin = int(y_center - 0.5*float(box[3])*img_h)
    xmax = int(x_center + 0.5*float(box[2])*img_w)
    ymax = int(y_center + 0.5*float(box[3])*img_h)
    boxes = [xmin,ymin,xmax,ymax]
    return boxes

'''
以下三类合并函数可以根据自己数据集的特点进行改写
大致思路是找到框后，通过计算框之间的交并比和交面比判断是否属于一个目标
比如person类包含人身、人头、吸烟等标签，可以将人身放在一起，其他放在一起
将交并比最大的认为同属一个person，然后计算几个框最大外接矩形得到新标签的框坐标信息
'''
def person_hebin(label_path,weights_img,height_img):
    all_heads, all_bodys = [],[]
    xml_label = []
    with open(label_path, 'r') as f:
        for j in f.readlines():
            j = j.strip().split(' ')
            if classes[int(j[0])] in ['wcgz','gzzc']:
                body = yolo2voc(j[1:],weights_img,height_img)
                all_bodys.append(body)							# 分别统计对应类并转换为voc格式
            elif classes[int(j[0])] in ['wcaqm','xy','aqmzc']:
                head = yolo2voc(j[1:],weights_img,height_img)
                all_heads.append(head)

        if len(all_heads)==0 and len(all_bodys)!=0:				#如果没有head,有几个body就生成几个框
            xml_label = all_bodys
        elif len(all_bodys)==0 and len(all_heads)!=0:			#如果没有body,有几个head就生成几个框
            xml_label = all_heads
        elif len(all_heads)==0 and len(all_bodys)==0:
            pass
		'''
		head和body都有，依次取一个body,分别计算heads中与其交并比，
		交并比最大的认为是一个person,保存其最小外接矩形的信息
		'''
        else:													
            for body in all_bodys:
                area_ratios = 0
                for head in all_heads:
                    iou,area_ratio,min_rect = Iou(head,body)
                    area_ratios = max(area_ratios,area_ratio)
                if 0<area_ratios<1:
                    xml_label.append(min_rect)
                    # all_heads.remove(head)
                else: 
                    xml_label.append(body)
                    # all_heads.remove(head)
            # print(xml_label)
    return xml_label

def bj_hebin(label_path,weights_img,height_img):
    all_bjbp, all_bjds, all_bjwk = [],[],[]
    xml_label = []
    # 读取对应.txt文件
    with open(label_path, 'r') as f:
        for j in f.readlines():
            j = j.strip().split(' ')
            if classes[int(j[0])] == 'bjwk_ps':
                bjwk = yolo2voc(j[1:],weights_img,height_img)
                all_bjwk.append(bjwk)
            elif classes[int(j[0])] in ['bj_bpmh','bj_bpps','bj_bpzc']:
                bjbp = yolo2voc(j[1:],weights_img,height_img)
                all_bjbp.append(bjbp)
            elif classes[int(j[0])] in ['bjdsyc','bjdszc']:
                bjds = yolo2voc(j[1:],weights_img,height_img)
                all_bjds.append(bjds)
        if len(all_bjwk)==0:
            if len(all_bjbp)==0 and len(all_bjds)==0:
                pass
            elif len(all_bjbp)==0 and len(all_bjds)!=0:
                xml_label = all_bjds
            elif len(all_bjbp)!=0 and len(all_bjds)==0:
                xml_label = all_bjbp
            else:
                for bjbp in all_bjbp:
                    area_ratios = 0
                    for bjds in all_bjds:
                        iou,area_ratio,min_rect = Iou(bjds,bjbp)
                        area_ratios = max(area_ratios,area_ratio)
                    if 0<area_ratios<1:
                        xml_label.append(min_rect)
                    else: 
                        xml_label.append(bjbp)
        else:
            if len(all_bjbp)!=0 and len(all_bjds)==0:
                for bjwk in all_bjwk:
                        area_ratios = 0
                        for bjbp in all_bjbp:
                            iou,area_ratio,min_rect = Iou(bjbp,bjwk)
                            area_ratios = max(area_ratios,area_ratio)
                        if 0<area_ratios<1:
                            xml_label.append(min_rect)
                            # all_heads.remove(head)
                        elif area_ratios==1:
                            xml_label.append(bjwk)
                            # all_heads.remove(head)
                        else: 
                            xml_label.append(bjbp)
            elif len(all_bjbp)==0 and len(all_bjds)!=0:
                    for bjwk in all_bjwk:
                        area_ratios = 0
                        for bjds in all_bjds:
                            iou,area_ratio,min_rect = Iou(bjds,bjwk)
                            area_ratios = max(area_ratios,area_ratio)
                        if 0<area_ratios<1:
                            xml_label.append(min_rect)
                            # all_heads.remove(head)
                        elif area_ratios==1:
                            xml_label.append(bjwk)
                            # all_heads.remove(head)
                        else: 
                            xml_label.append(bjds)
            else:
                for bjwk in all_bjwk:
                    area_ratios1 = 0
                    area_ratios2 = 0
                    for bjbp in all_bjbp:
                        iou,area_ratio,min_rect = Iou(bjbp,bjwk)
                        area_ratios1 = max(area_ratios1,area_ratio)
                    for bjds in all_bjds:
                        iou,area_ratio,min_rect = Iou(bjds,min_rect)
                        area_ratios2 = max(area_ratios2,area_ratio)
                    if 0<area_ratios2<1:
                        xml_label.append(min_rect)
                        # all_heads.remove(head)
                    elif area_ratios2==1:
                        xml_label.append(bjwk)
                        # all_heads.remove(head)
                    else: 
                        xml_label.append(bjbp)
        # print(xml_label)
    return xml_label

def hxq_hebin(label_path,weights_img,height_img):
    all_gj, all_gjt, all_yf = [],[],[]
    xml_label = []
    with open(label_path, 'r') as f:
        for j in f.readlines():
            j = j.strip().split(' ')
            if classes[int(j[0])] in ['hxq_gjbs','hxq_gjzc']:
                gj = yolo2voc(j[1:],weights_img,height_img)
                all_gj.append(gj)
            elif classes[int(j[0])] in ['hxq_gjtps','ywzt_yfyc']:
                gjt = yolo2voc(j[1:],weights_img,height_img)
                all_gjt.append(gjt)
            elif classes[int(j[0])] in ['ywzt_yfyc']:
                yf = yolo2voc(j[1:],weights_img,height_img)
                all_yf.append(yf)

        if len(all_gj)==0:
            if len(all_gjt)==0 and len(all_yf)==0:
                pass
            elif len(all_gjt)==0 and len(all_yf)!=0:
                xml_label = all_yf
            elif len(all_gjt)!=0 and len(all_yf)==0:
                xml_label = all_gjt
            else:
                for gjt in all_gjt:
                    area_ratios = 0
                    for yf in all_yf:
                        gjt_kd = [gjt[0],gjt[1],gjt[2],gjt[3]+0.5*(gjt[3]-gjt[1])]
                        iou_kd = Iou(yf,gjt_kd)
                        if iou_kd>0:
                            iou,area_ratio,min_rect = Iou(yf,gjt)
                            area_ratios = max(area_ratios,area_ratio)
                    if 0<area_ratios<1:
                        xml_label.append(min_rect)
                    else: 
                        xml_label.append(gjt)
        else:
            if len(all_gjt)!=0 and len(all_yf)==0:
                for gj in all_gj:
                        area_ratios = 0
                        for gjt in all_gjt:
                            iou,area_ratio,min_rect = Iou(gjt,gj)
                            area_ratios = max(area_ratios,area_ratio)
                        if 0<area_ratios<1:
                            xml_label.append(min_rect)
                        else: 
                            xml_label.append(gj)
            elif len(all_gjt)==0 and len(all_yf)!=0:
                for gj in all_gj:
                    area_ratios = 0
                    for yf in all_yf:
                        gj_kd = [gj[0],gj[1],gj[2],gj[3]+0.5*(gj[3]-gj[1])]
                        iou_kd = Iou(yf,gj_kd)
                        if iou_kd>0:
                            iou,area_ratio,min_rect = Iou(yf,gj)
                            area_ratios = max(area_ratios,area_ratio)
                    if 0<area_ratios<1:
                        xml_label.append(min_rect)
                    else: 
                        xml_label.append(gj)
            elif len(all_gjt)==0 and len(all_yf)==0:
                xml_label = all_gj
            else:
                for gj in all_gj:
                    area_ratios1 = 0
                    area_ratios2 = 0
                    rect = []
                    for gjt in all_gjt:
                        iou,area_ratio,min_rect = Iou(gjt,gj)
                        area_ratios1 = max(area_ratios1,area_ratio)
                    if 0<area_ratios<1:
                        rect = min_rect
                    elif area_ratios==1:
                        rect = gj
                    else: 
                        rect = yf
                    for yf in all_yf:
                        rect_kd = [rect[0],rect[1],rect[2],rect[3]+0.5*(rect[3]-rect[1])]
                        iou_kd = Iou(yf,rect_kd)
                        if iou_kd>0:
                            iou,area_ratio,min_rect = Iou(yf,rect)
                            area_ratios2 = max(area_ratios2,area_ratio)
                    if 0<area_ratios2<1:
                        xml_label.append(min_rect)
                        # all_heads.remove(head)
                    else: 
                        xml_label.append(rect)
        # print(xml_label)
    return xml_label


#def create_new_label(image_dst,label_dst,new_label_dst):
    # 读取对应标签集
    label = ['person','bj','hxq']
    image_path = image_dst+label[1]
    # 获取文件名
    image_name = os.listdir(image_path)
    xml_label = [] 
	
    for i in image_name:
        print('read image:',i)
        img = cv2.imread(os.path.join(image_path, i))
        height_img, weights_img, depth_img = img.shape
        label_path = os.path.join(label_dst,i.split('.jpg')[0])+'.txt'
        xml_label_person = person_hebin(label_path,weights_img,height_img,i,new_label_dst)
        xml_label_bj = bj_hebin(label_path,weights_img,height_img,i,new_label_dst)
        xml_label_hxq = hxq_hebin(label_path,weights_img,height_img,i,new_label_dst)
        xml_label = [xml_label_person,xml_label_bj,xml_label_hxq]
        print(xml_label)

        # 创建xml标签文件中的标签
        xmlBuilder = Document()
        # 创建annotation标签，也是根标签
        annotation = xmlBuilder.createElement("annotation")
        # 给标签annotation添加一个子标签
        xmlBuilder.appendChild(annotation)

        folder = xmlBuilder.createElement("folder")
        folderContent = xmlBuilder.createTextNode(image_dst.split('/')[-1]+label[1])  # 标签内存
        folder.appendChild(folderContent)  # 把内容存入标签
        annotation.appendChild(folder)   # 把存好内容的folder标签放到 annotation根标签下

        filename = xmlBuilder.createElement("filename")
        filenameContent = xmlBuilder.createTextNode(i.split('.')[0] + '.jpg')  # 标签内容
        filename.appendChild(filenameContent)
        annotation.appendChild(filename)

        size = xmlBuilder.createElement("size")
        width = xmlBuilder.createElement("width")  
        widthContent = xmlBuilder.createTextNode(str(weights_img))
        width.appendChild(widthContent)
        size.appendChild(width)  

        height = xmlBuilder.createElement("height")  
        heightContent = xmlBuilder.createTextNode(str(height_img))  
        height.appendChild(heightContent)
        size.appendChild(height) 

        depth = xmlBuilder.createElement("depth") 
        depthContent = xmlBuilder.createTextNode(str(depth_img))
        depth.appendChild(depthContent)
        size.appendChild(depth) 
        annotation.appendChild(size)  

        for object_info in xml_label:
            for object_info_box in object_info:
            # 开始创建标注目标的label信息的标签
                object = xmlBuilder.createElement("object")  
                imgName = xmlBuilder.createElement("name")  
                imgNameContent = xmlBuilder.createTextNode(label[1])
                imgName.appendChild(imgNameContent)
                object.appendChild(imgName)  

                pose = xmlBuilder.createElement("pose")
                poseContent = xmlBuilder.createTextNode("Unspecified")
                pose.appendChild(poseContent)
                object.appendChild(pose) 

                truncated = xmlBuilder.createElement("truncated")
                truncatedContent = xmlBuilder.createTextNode("0")
                truncated.appendChild(truncatedContent)
                object.appendChild(truncated)

                difficult = xmlBuilder.createElement("difficult")
                difficultContent = xmlBuilder.createTextNode("0")
                difficult.appendChild(difficultContent)
                object.appendChild(difficult)

                bndbox = xmlBuilder.createElement("bndbox")
                xmin = xmlBuilder.createElement("xmin")  
                xminContent = xmlBuilder.createTextNode(str(object_info_box[0]))
                xmin.appendChild(xminContent)
                bndbox.appendChild(xmin)
                
                ymin = xmlBuilder.createElement("ymin") 
                yminContent = xmlBuilder.createTextNode(str(object_info_box[1]))
                ymin.appendChild(yminContent)
                bndbox.appendChild(ymin)

                xmax = xmlBuilder.createElement("xmax")  
                xmaxContent = xmlBuilder.createTextNode(str(object_info_box[2]))
                xmax.appendChild(xmaxContent)
                bndbox.appendChild(xmax)

                ymax = xmlBuilder.createElement("ymax") 
                ymaxContent = xmlBuilder.createTextNode(str(object_info_box[3]))
                ymax.appendChild(ymaxContent)
                bndbox.appendChild(ymax)

                object.appendChild(bndbox)
                annotation.appendChild(object)  
            f = open(os.path.join(new_label_dst, i.split('.')[0]+'.xml'), 'w')
            xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()

class YOLO2VOCConvert:
    def __init__(self, label_src, label_dst, new_label_dst, image_src, image_dst):
        self.txts_src_path = label_src   
        self.txts_dst_path = label_dst
        self.new_label_path = new_label_dst   
        self.imgs_src_path = image_src  
        self.imgs_dst_path = image_dst
        self.classes = ['person','bj','hxq']

    def read(self):
        image_list = glob.glob(self.imgs_src_path+"*.jpg")
        print('一共有%d张图'%len(image_list))
        return image_list

    def copy(self,image_path,label_path,image_base_name,label_name):
        # image_base_name = os.path.basename(image_path)
        # label_name = image_base_name.split('.jpg')[0]+'.txt'
        # label_path = os.path.join(self.txts_src_path,label_name)
        # if os.path.exists(label_path) is True:
        with open(label_path, 'r') as f:
            person_num,bj_num,hxq_num = 0,0,0
            for i in f.readlines():
                index = i.split(' ')[0]
                try: 
                    if classes[int(index)] in person:
                        person_num += 1
                        continue
                    elif classes[int(index)] in bj:
                        bj_num += 1
                        continue
                    elif classes[int(index)] in hxq:
                        hxq_num += 1
                        continue
                except IndexError as error:
                    continue
            if (person_num+bj_num+hxq_num)>0:
                shutil.copy(image_path, os.path.join(self.imgs_dst_path,image_base_name))
                shutil.copy(label_path, os.path.join(self.txts_dst_path,label_name))
                print('copy down,person:{},bj:{},hxq:{}'.format(person_num,bj_num,hxq_num))

        # return os.path.join(label_dst,label_name)
        return label_name,(person_num+bj_num+hxq_num)
	
	#	将上面得到的框信息写进xml文件
    def yolo_voc(self, new_label_name):
        xmls_path = os.path.join(self.new_label_path,new_label_name.split('.txt')[0])+'.xml'
        if not os.path.exists(xmls_path):
            # f = open(xmls_path, 'w')
			
			# 得到一张图合并后的全部框信息
            img = cv2.imread(image_path)
            height_img, weights_img, depth_img = img.shape
            label_path = os.path.join(self.txts_src_path,new_label_name)
            xml_label_person = person_hebin(label_path,weights_img,height_img)
            xml_label_bj = bj_hebin(label_path,weights_img,height_img)
            xml_label_hxq = hxq_hebin(label_path,weights_img,height_img)
            xml_label = [xml_label_person,xml_label_bj,xml_label_hxq]
            print(xml_label)

            # 创建xml标签文件中的标签
            xmlBuilder = Document()
            # 创建annotation标签，也是根标签
            annotation = xmlBuilder.createElement("annotation")
            # 给标签annotation添加一个子标签
            xmlBuilder.appendChild(annotation)

            folder = xmlBuilder.createElement("folder")
            folderContent = xmlBuilder.createTextNode(self.imgs_dst_path.split('/')[-1])  # 标签内存
            folder.appendChild(folderContent)  # 把内容存入标签
            annotation.appendChild(folder)   # 把存好内容的folder标签放到 annotation根标签下

            filename = xmlBuilder.createElement("filename")
            filenameContent = xmlBuilder.createTextNode(new_label_name.split('.')[0] + '.jpg')  # 标签内容
            filename.appendChild(filenameContent)
            annotation.appendChild(filename)

            size = xmlBuilder.createElement("size")
            width = xmlBuilder.createElement("width")  
            widthContent = xmlBuilder.createTextNode(str(weights_img))
            width.appendChild(widthContent)
            size.appendChild(width)  

            height = xmlBuilder.createElement("height")  
            heightContent = xmlBuilder.createTextNode(str(height_img))  
            height.appendChild(heightContent)
            size.appendChild(height) 

            depth = xmlBuilder.createElement("depth") 
            depthContent = xmlBuilder.createTextNode(str(depth_img))
            depth.appendChild(depthContent)
            size.appendChild(depth) 
            annotation.appendChild(size)  

            for i,object_info in enumerate(xml_label):
                for object_info_box in object_info:
                # 开始创建标注目标的label信息的标签
                    object = xmlBuilder.createElement("object")  
                    imgName = xmlBuilder.createElement("name")  
                    imgNameContent = xmlBuilder.createTextNode(self.classes[i])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)  

                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose) 

                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    bndbox = xmlBuilder.createElement("bndbox")
                    xmin = xmlBuilder.createElement("xmin")  
                    xminContent = xmlBuilder.createTextNode(str(object_info_box[0]))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    
                    ymin = xmlBuilder.createElement("ymin") 
                    yminContent = xmlBuilder.createTextNode(str(object_info_box[1]))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)

                    xmax = xmlBuilder.createElement("xmax")  
                    xmaxContent = xmlBuilder.createTextNode(str(object_info_box[2]))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)

                    ymax = xmlBuilder.createElement("ymax") 
                    ymaxContent = xmlBuilder.createTextNode(str(object_info_box[3]))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)  
                f = open(xmls_path, 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()
                


if __name__ == '__main__':
    image_src = ".\\datasets\\data_tmp_game\\data_tmp_game\\"	#原始图片目录
    image_dst = ".\\datasets\\dianli17\\JPEGimages\\"			#图片保存目录
    label_src = ".\dianli17_yolo_txt/"							#原始标签目录
    label_dst = ".datasets\\dianli17\\txt\\"					#保存标签目录
    new_label_dst = ".datasets\\dianli17\\xml\\"				#转换为xml格式后的保存路径

    yolo_voc = YOLO2VOCConvert(label_src,label_dst,new_label_dst,image_src,image_dst)
    now_label_list = glob.glob(label_dst+"*.txt")
    label_list = glob.glob(label_src+"*.txt")
    listA,listB=[],[]
    for now_label in now_label_list:
        listA.append(os.path.basename(now_label))
    print(len(listA))
    for label in label_list:
        listB.append(os.path.basename(label))
    print(len(listB))
    label_list1 = list(set(listB)-set(listA))		#获得未进行转换的标签列表
    print(len(label_list1))

    for label_path in label_list1:
        label_base_name = label_path
        image_name = label_base_name.split('.txt')[0]+'.jpg'
        if os.path.exists(os.path.join(image_src,image_name)) is True:
            print('now read ',label_base_name)
            image_path = os.path.join(image_src,image_name)
            label_path = os.path.join(label_src, label_base_name)
            new_label_name,object_num = yolo_voc.copy(image_path,label_path,image_name,label_base_name)
            if object_num>0:		#图中三类数量大于0才进行转换
                yolo_voc.yolo_voc(new_label_name)
    # now_image_list = glob.glob(image_dst+"*.jpg")
    # listA,listB=[],[]
    # for now_image in now_image_list:
    #     listA.append(os.path.basename(now_image))
    # print(len(listA))
    # image_list = yolo_voc.read()
    # for image in image_list:
    #     listB.append(os.path.basename(image))
    # image_list1 = list(set(listB)-set(listA))
    # print(len(image_list1))
    # time.sleep(5)
    # for image_path in image_list1:
    #     image_base_name = image_path
    #     label_name = image_base_name.split('.jpg')[0]+'.txt'
    #     # label_path = os.path.join(label_src,label_name)
    #     if os.path.exists(os.path.join(label_src,label_name)) is True:
    #         print('now read ',label_name)
    #         label_path = os.path.join(label_src,label_name)
    #         image_path = os.path.join(image_src,image_base_name)
    #         new_label_name,object_num = yolo_voc.copy(image_path,label_path,image_base_name,label_name)
    #         if object_num>0:
    #             yolo_voc.yolo_voc(new_label_name)
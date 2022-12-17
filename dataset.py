import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ic_dataset(Dataset):
    def __init__(self, txt_path, img_path,  transform=None):
        super(ic_dataset, self).__init__()
        self.txt_lines = self.readlines(txt_path)
        self.img_path = img_path
        self.transform = transform
        self.img_info_list = self.parse_lines(self.txt_lines)
    
    def parse_lines(self,lines):
        image_info_list = []
        for line in lines:
            line_split = line.strip().split("  ")
            img_name = line_split[0]
            img_label = line_split[1]
            image_info_list.append((img_name,img_label))      
        return image_info_list  
    
    def readlines(self,txt_path):
        f = open(txt_path,'r')
        lines = f.readlines()
        f.close()
        return lines     
    
    def __getitem__(self,index):
        imgName, imgLabel= self.img_info_list[index]
        oriImgPath = os.path.join(self.img_path, imgName)
        img = Image.open(oriImgPath).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(float(imgLabel))
        return img,label,imgName
    
    def __len__(self):
        return len(self.img_info_list) 
import cv2 
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
def ssw(img,scale=500,sigma=0.9,min_size=10):
    img_lbl,regions=selectivesearch.selective_search(img,scale=scale,sigma=sigma,min_size=min_size)
    candidates =set()
    for r in regions:
        # 重复的不要
        if r['rect'] in candidates:
            continue
        # 太小和太大的不要
        if r['size'] < 200 or r['size']>22500:
            continue
        x, y, w, h = r['rect']
        # 太不方的不要
        if w / h > 2 or h / w > 2:
            continue
        candidates.add(r['rect'])
        ##('len(candidates)', 34) 一次过滤后剩余34个窗
    #2)第二次过滤 大圈套小圈的目标 只保留大圈
    num_array=[]
    for i in candidates:
        if len(num_array)==0:
            num_array.append(i)
        else:
            content=False
            replace=-1
            index=0
        for j in num_array:
    ##新窗口在小圈 则滤
            if i[0]>=j[0] and i[0]+i[2]<=j[0]+j[2]and i[1]>=j[1] and i[1]+i[3]<=j[1]+j[3]:
                content=True
                break
            ##新窗口不在小圈 而在老窗口外部 替换老窗口
            elif i[0]<=j[0] and i[0]+i[2]>=j[0]+j[2]and i[1]<=j[1] and i[1]+i[3]>=j[1]+j[3]:
                replace=index
                break
                index+=1
            if not content:
                if replace>=0:
                    num_array[replace]=i
                else:
                    num_array.append(i)
            #窗口过滤完之后的数量
    num_array=set(num_array)
    return num_array

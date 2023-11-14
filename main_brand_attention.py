from Brand_Attention_module.Brand_attention_module import brand_attention_calc
import time
from glob import glob

tmap = glob("test_images/text/*_tmap.png")
data=[]
for t in tmap:
    x=t.split("_tmap")[0]+".png"
    data.append(x)

for i in range(len(data)):
    img_path = data[i]
    tmap_path = tmap[i]
    score = brand_attention_calc(img_path, tmap_path)
    print("*****************************")
    print(img_path)
    print(score)

# img_path = "test_images/C1.png"
# tmap = "test_images/C1_t.png"

# start = time.time()
# brand_attention_calc(img_path,tmap)
# print(time.time() - start)
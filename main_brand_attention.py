from Brand_Attention_module.Brand_attention_module import brand_attention_calc
import time
img_path = "test_images/test.jpg"
tmap = "test_images/test_tmap.jpg"

start = time.time()
brand_attention_calc(img_path,tmap)
print(time.time() - start)
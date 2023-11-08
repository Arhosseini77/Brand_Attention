from Brand_Attention_module.Brand_attention_module import brand_attention_calc
import time
img_path = "test_images/C1.png"
tmap = "test_images/C1_t.png"

start = time.time()
brand_attention_calc(img_path,tmap)
print(time.time() - start)
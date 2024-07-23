import numpy as np
from PIL import Image

from siamese import Siamese
import time

if __name__ == "__main__":
    model = Siamese()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        start_time = time.time()
        # 在此处写入您要计算执行时间的代码
        probability = model.detect_image(image_1,image_2)
        end_time = time.time()

        execution_time = end_time - start_time
        # print(f"代码执行时间: {execution_time} 秒")
        print(probability)


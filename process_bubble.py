import time
import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict
import sys

def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx
        print(f"question {idx} {score[0]} {score[1]}")
        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = 'choice'
            results[question + 1].append(chosed_answer)
          
        if score[0] > 0.9:  # choiced confidence score > 0.9
            chosed_answer = 'unchoice'
            results[question + 1].append(chosed_answer)

    return results


if __name__ == '__main__':
    start_time = time.time()
    
    image_paths = sys.argv[1].split(',')
    
    list_bubble_img = []
    for idx, path in enumerate(image_paths):
        bubble_img = cv2.imread(path)
        #print(f"Kích thước ảnh gốc {idx}: {bubble_img.shape}")
        bubble_img = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
        bubble_img = cv2.threshold(bubble_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        bubble_img = cv2.resize(bubble_img, (28, 28), cv2.INTER_AREA)
        #print(f"Kích thước sau khi resize {idx}: {bubble_img.shape}")
        bubble_img = bubble_img.reshape((28, 28, 1))
        #cv2.imwrite('1.jpg',bubble_img)
        list_bubble_img.append(bubble_img)
    
    result = get_answers(list_bubble_img);
    
    end_time = time.time()  # Kết thúc đo thời gian
    execution_time = end_time - start_time
    print(f"Thời gian thực thi: {execution_time} giây")
    
    print(result)

    start_time2 = time.time()

    end_time2 = time.time()
    execution_time2 = end_time2 - start_time2  # Tính thời gian thực thi
    print(f"Thời gian thực thi 2: {execution_time2} giây")

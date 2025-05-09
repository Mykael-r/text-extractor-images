import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


print("GPU disponível:", torch.cuda.is_available())

reader = easyocr.Reader(['pt', 'en'], gpu=torch.cuda.is_available())

image_path = 'images/etiqueta_1.jpg'

img = cv2.imread(image_path)

results = reader.readtext(img)

for (bbox, text, prob) in results:
    print(f"Texto: {text}) - Probabilidade: {prob:.2f}")

for (bbox, text, prob) in results:
    cv2.polylines(img,[np.int32(bbox)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img, text, (int(bbox[0][0]), int(bbox[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Texto Detectado")
plt.show()
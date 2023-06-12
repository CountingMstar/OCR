"""
https://machinelearningknowledge.ai/easyocr-python-tutorial-with-examples/
"""

import easyocr
import cv2
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import cv2
print(cv2.__version__)

# img = cv2.imread('1.png')
img = cv2.imread('text_image.png')
# plt.imshow(img)
# cv2.imshow(img)

reader = easyocr.Reader(['en'])

# 단어 단위로 출력
result = reader.readtext(img, detail = 0)
print(result)

# 문장 단위로 출력
result = reader.readtext(img, detail = 0, paragraph = True)
print(result)

# 각 단어별 detail(바운딩 박스 위치, 단어, 정확도) 출력
result = reader.readtext(img, detail = 1, paragraph = False)
print(result)

# 바운딩 박스를 그려줌
# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(img)

for (coord, text, prob) in result:
    (topleft, topright, bottomright, bottomleft) = coord
    tx,ty = (int(topleft[0]), int(topleft[1]))
    bx,by = (int(bottomright[0]), int(bottomright[1]))
    x = abs(tx - bx)
    y = abs(ty - by)
    # patches.rectangle(img, (tx,ty), (bx,by), (0, 0, 255), 2)

    # Create a Rectangle patch
    rect = patches.Rectangle((tx, ty), x, y, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

print('2222222222222222222')
# ax.imshow(img)
# plt.imshow(img)
plt.savefig('10.png')
plt.show()

# plot clear
plt.cla()
plt.show()
print('33333333333333333333333')


sentence = ''
k = 0.5
mask = False

for r in result:
    if r[2] > k:
        print(r[1])
        sentence += r[1] + ' '
    else:
        sentence +=  '[MASK] '
        mask = True

print('4444444444444444444')
print(sentence)


# Load the pre-trained model for masked language modeling
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Define the input sentence with masked words
if mask == True:

    input_sentence = sentence

    # Generate predictions for the masked word
    predictions = fill_mask(input_sentence)

    # Print the predicted words and their probabilities
    for prediction in predictions:
        predicted_word = prediction["token_str"]
        probability = prediction["score"]
        print(f"Predicted word: {predicted_word}, Probability: {probability:.4f}")
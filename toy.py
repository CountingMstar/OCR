"""
사진 자르기
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import pickle
from PIL import Image, ImageDraw, ImageFont

i = 4
image = cv2.imread('data/noise_text/noise_image' + str(i) + '.png')

reader = easyocr.Reader(['en'])
result = reader.readtext(image, detail = 1, paragraph = False)

a = 4
print(result[a])

coord, text, prob = result[a][0], result[a][1], result[a][2]
print(coord)
print(text)
print(prob)

(topleft, topright, bottomright, bottomleft) = coord
tx,ty = (int(topleft[0]), int(topleft[1]))
bx,by = (int(bottomright[0]), int(bottomright[1]))
x = abs(tx - bx)
y = abs(ty - by)

region = image[0:100, tx:tx+x]
cv2.imwrite('yes.png', region)


##############################################

texts = ['singed', 'singing', 'yesyes']

for i in range(len(texts)):

    text = texts[i]

    # Set the font properties
    font_size = 100
    font_color = (0, 0, 0)  # RGB color tuple
    font_path = "/home/moonstar/python/NLP/OCR/Roboto/Roboto-Regular.ttf"  # Replace with the path to your desired font file

    # Set the image size based on the text length and font size
    image_width = len(text) * int(font_size/2)
    image_height = font_size
    image_size = (image_width, image_height)

    # Create a blank image with a white background
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)

    # Calculate the position to center the text
    text_x = (image_width - text_bbox[2]) // 2
    text_y = (image_height - text_bbox[3]) // 2

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=font, fill=font_color)

    # Save the image
    image.save("data/candidate_text/candidate_image" + str(i) + ".png")


################################################

imageA = cv2.imread('yes.png')
i = 2
imageB = cv2.imread("data/candidate_text/candidate_image" + str(i) + ".png")


# print(imageA)
print(imageA.shape)

# print(imageB)
print(imageB.shape)


# compare_images(imageA, imageB, title)

import torch
from sklearn.metrics.pairwise import cosine_similarity

"""
빼기
"""
# imageA = torch.tensor(imageA)
# imageB = torch.tensor(imageB)

# imageA = imageA.reshape(1, -1)
# imageB = imageB.reshape(1, -1)

# print(imageA.shape)
# print(imageB.shape)

# A = imageA.shape[1]
# B = imageB.shape[1]

# tmp = [A, B]
# small = min(tmp)
# print(small)

# imageA = imageA[0][:small]
# imageB = imageB[0][:small]

# imageA = imageA.reshape(1, -1)
# imageB = imageB.reshape(1, -1)

# print(imageA)
# print(imageB)
# print(imageA.shape)
# print(imageB.shape)


"""
패딩
"""
imageA = torch.tensor(imageA)
imageB = torch.tensor(imageB)

imageA = imageA.reshape(1, -1)
imageB = imageB.reshape(1, -1)

print(imageA.shape)
print(imageB.shape)

A = imageA.shape[1]
B = imageB.shape[1]

images = [imageA, imageB]
tmp = [A, B]
big = max(tmp)
big_index = tmp.index(big)
small = min(tmp)
small_index = tmp.index(small)

print(big)
print(big_index)

imageA = imageA[0][:big]
imageB = imageB[0][:big]

add = torch.tensor([[0 for i in range(abs(A-B))]])

images[small_index] = torch.cat((images[small_index], add), dim=-1)

imageA = images[0]
imageB = images[1]

print(imageA)
print(imageB)
print(imageA.shape)
print(imageB.shape)


sim = cosine_similarity(imageA, imageB)
print(sim)
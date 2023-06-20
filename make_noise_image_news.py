import numpy as np
from PIL import Image
import easyocr
import cv2
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pickle


# import cv2
print(cv2.__version__)

data_len = 1499
randoms = []

for i in range(data_len):
    # img = cv2.imread('1.png')
    img = cv2.imread('data/original_text_news/text_image' + str(i) + '.png')
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
    print(len(result))
    print(result)
    print(len(result)-1)

    start = 0
    end = len(result)-1
    rand = random.randrange(start, end)
    randoms.append(rand)
    print('77777777777')
    print(rand)
    print(result[rand])
    print(result[rand][0][0][0])



    # 바운딩 박스를 그려줌
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 1))

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
    # plt.savefig('10.png')
    # plt.show()



    # Load the image
    image_path = "data/original_text_news/text_image" + str(i) + ".png"  # Replace with the path to your image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Define the region of interest (ROI) using a mask
    roi_mask = np.zeros_like(image_array, dtype=np.uint8)
    start_x, start_y = result[rand][0][0][0], result[rand][0][0][1]  # Coordinates of the top-left corner of the ROI
    end_x, end_y = result[rand][0][2][0], result[rand][0][2][0]  # Coordinates of the bottom-right corner of the ROI
    mid_x, mid_y = int((end_x - start_x) / 2) + start_x, int((end_y - start_y) / 2) + start_y

    start_y = 0
    end_y = 40

    # start_y = 50
    # end_y = 100

    start_x = start_x + 10
    end_x = end_x - 10

    # roi_mask[start_y:end_y, start_x:end_x] = 1
    roi_mask[0:20, start_x:end_x] = 1
    roi_mask[40:60, start_x:end_x] = 1

    # Set the strength of the noise
    noise_strength = 5

    # Generate random noise for the ROI
    roi_noise = np.random.normal(loc=0, scale=noise_strength, size=image_array.shape[:2])
    roi_noise = np.expand_dims(roi_noise, axis=-1)  # Add a channel dimension for broadcasting
    roi_noise = roi_noise.astype(np.uint8)

    # Apply the noise to the ROI only
    noisy_image_array = np.where(roi_mask, np.clip(image_array + roi_noise, 0, 255), image_array)

    # Convert the NumPy array back to an image
    noisy_image = Image.fromarray(noisy_image_array)

    # Save the noisy image
    noisy_image.save("data/noise_text_news/noise_image" + str(i) + ".png")
    # noisy_image.show()



data = randoms

file_path = "data.pickle"  # Specify the file path and name
with open(file_path, "wb") as file:
    # Write the data to the file using pickle.dump()
    pickle.dump(data, file)
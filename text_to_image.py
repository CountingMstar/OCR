from PIL import Image, ImageDraw, ImageFont
import nltk

nltk.download('punkt')  # Download the necessary resource (only required once)

# 텍스트 파일 불러오기
file_path = "/home/moonstar/python/NLP/OCR/data/train.txt"  # Replace with the actual path to your text file

with open(file_path, 'r') as file:
    data = file.read()

new_data = []
for i in data:
    if i == ' ':
        new_data.append('   ')
    else:
        new_data.append(i)

new_data = ''.join(new_data)

# Tokenize the sentence
# tokens = nltk.word_tokenize(sentence)
sentences = nltk.sent_tokenize(new_data)


# Define the text to be converted into an image
# text = "Hello, World!"
print('==============')
# text = sentences[0][:50]
texts = sentences
print(texts)

for i in range(len(texts)):

    text = texts[i]

    if i == 5 or i == 9:
        print('3333333333333333333')
        print(text)
        print(len(text))

    if i == 11:
        print(len(text))
        print(text)


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
    image.save("data/original_text/text_image" + str(i) + ".png")
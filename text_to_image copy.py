from PIL import Image, ImageDraw, ImageFont

# Define the text to be converted into an image
text = "Hello, World!"

# Set the font properties
font_size = 30
font_color = (0, 0, 0)  # RGB color tuple
font_path = "/home/moonstar/python/NLP/OCR/Roboto/Roboto-Regular.ttf"  # Replace with the path to your desired font file

# Set the image size based on the text length and font size
image_width = len(text) * font_size
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
image.save("text_image.png")
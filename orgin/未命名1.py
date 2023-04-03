import cv2
from PIL import Image
import sys


def cut_image(image):
    width, height = image.size
    print(width, height)
    item_width = int(width / 800)
    box_list = []
    # (left, upper, right, lower)
    for i in range(800):#0.01s为一段，8s=800个0.01s
        box = (i*item_width,0,(i+1)*item_width,height)
        box_list.append(box)

    image_list = [image.crop(box) for box in box_list] #crop用于切割图片
    return image_list

def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('../img/ing/'+str(index) + '.png', 'PNG')
        index += 1

if __name__ == '__main__':
    file_path = "../img/ii.png"
    image = Image.open(file_path)
    #image.show()
    image_list = cut_image(image)
    save_images(image_list)

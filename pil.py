from PIL import Image
im = Image.open('boxer-starter.jpg')
im.quantize(256)
im.show()
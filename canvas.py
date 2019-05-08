from tkinter import *
from tkinter.colorchooser import askcolor
import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
import neuralNet
import utils
import data


class Paint(object):

    DEFAULT_PEN_SIZE = 16.0
    DEFAULT_COLOR = 'black'

    def __init__(self, net):
        self.root = Tk()

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.save_button = Button(self.root, text='save', command=self.save_drawing)
        self.save_button.grid(row=0, column=2)

        self.choose_size_button = Scale(self.root, from_=16, to=20, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=560, height=560)
        self.c.grid(row=1, columnspan=5)
        self.image1 = PIL.Image.new("RGB", (560, 560), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.net = net
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=not self.eraser_on)

    def save_drawing(self):
        filename = "image.bmp"
        js = np.ravel(self.image1.convert('L').resize((28,28)))
        js = abs(js - 255).reshape((1,784))
        print(np.argmax(self.net.predict(js)[0]))
        self.activate_button(self.save_button)

    def activate_button(self, some_button, eraser_mode=False):
        some_button.config(relief=SUNKEN)
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=paint_color, width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    net = neuralNet.Network([784, 25, 10])
    x, y, test, y_t = data.load_data(1000, 150)
    y_d = utils.vectorized_result(y, 10)
    net.GD(x, y_d, 1, 150, 0.01)
    print(utils.get_accuracy(net.predict(test), y_t))
    Paint(net)
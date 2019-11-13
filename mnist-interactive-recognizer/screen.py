import pygame
import numpy as np


class DrawingScreen:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    MNIST_IM_DIMS = (28, 28)

    def __init__(self,
                 screen_dimensions_scale,
                 brush_radius=10,
                 scaled_down_dimentions=MNIST_IM_DIMS):
        w, h = scaled_down_dimentions
        dims = (w * screen_dimensions_scale, h * screen_dimensions_scale)
        self.scaled_down_dimentions = scaled_down_dimentions
        self.screen = pygame.display.set_mode(dims)
        self.__scale = screen_dimensions_scale
        self.__radius = brush_radius

    def dec_to_RGB(color):
        B = color % 256
        G = (color >> 8) % 256
        R = (color >> 16)
        return (R, G, B)

    def RGB_to_avarage_grayscale(color):
        (R, G, B) = color
        return (R + G + B) // 3

    def dec_to_grayscale(color):
        RGB = DrawingScreen.dec_to_RGB(color)
        grayscale = DrawingScreen.RGB_to_avarage_grayscale(RGB)
        return grayscale

    def erase(self):
        self.screen.fill(self.BLACK)

    def draw_line(self, start, end):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        steps = max(abs(delta_x), abs(delta_y))
        for i in range(steps):
            x = int(start[0] + ((i / steps) * delta_x))
            y = int(start[1] + ((i / steps) * delta_y))
            pygame.draw.circle(self.screen, self.WHITE, (x, y), self.__radius)

    def mainloop(self, on_mouse_button_up):
        draw = False
        last_position = (0, 0)

        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                self.erase()
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(self.screen, self.WHITE, event.pos,
                                   self.__radius)
                draw = True
            if event.type == pygame.MOUSEBUTTONUP:
                draw = False
                on_mouse_button_up(self.downscale())
            if event.type == pygame.MOUSEMOTION:
                if draw:
                    pygame.draw.circle(self.screen, self.WHITE, event.pos,
                                       self.__radius)
                    self.draw_line(event.pos, last_position)
                last_position = event.pos
            pygame.display.flip()

    def downscale(self):
        scr2d = np.matrix(pygame.surfarray.array2d(self.screen))
        grayscale = np.array(DrawingScreen.dec_to_grayscale(scr2d))
        transition_shape = (self.scaled_down_dimentions[0], self.__scale,
                            self.scaled_down_dimentions[1], self.__scale)
        downscaled = grayscale.reshape(transition_shape).mean(-1).mean(1)
        normalized = downscaled / 255
        return normalized.transpose()


if __name__ == '__main__':
    # show whole array when printing
    np.set_printoptions(threshold=np.inf)

    scale = 30
    brush_radius = 15
    scaled_down_dims = (10, 10)

    # print downscaled matrix on MOUSEBUTTONUP
    ds = DrawingScreen(scale, brush_radius, scaled_down_dims)
    ds.mainloop(lambda scr: print(scr))

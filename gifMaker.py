import imageio
import os


class GifAgent:
    def __init__(self):
        self.storage = []
        self.max_score = 0
        self.max_storage = []

    def store(self, img):
        self.storage.append(img)

    def commit(self, score, auto_output=False):
        if score > self.max_score:
            self.max_score = score
            self.max_storage = self.storage.copy()
            if auto_output:
                self.output()

        self.storage = []

    def output(self, name='max_score.gif'):
        if 'gif' not in os.listdir(os.getcwd()):
            os.mkdir('./gif')
        imageio.mimsave('./gif/'+name, self.max_storage, 'GIF', duration=0.03)

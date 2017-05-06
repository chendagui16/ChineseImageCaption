# @author: Dagui Chen
# @email: goblin_chen@163.com


class Config(object):
    def __init__(self):
        self.workspace = '/home/dagui/.keras/datasets/'
        self.fc2 = self.workspace + 'image_vgg19_fc2_feature_677004464.h5'
        self.caption = self.workspace + '{}.txt'

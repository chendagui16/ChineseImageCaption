# @ Author: Dagui Chen/ Xin Su
# @ Email: goblin_chen@163.com/suxin5987@qq.com
# @ Date: 2017-05-08/2017-05-19
# =====================================
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("workspace", "/home/dagui/.keras/datasets/", "The workspace dir path")
flags.DEFINE_string("feature_path", "image_vgg19_block5_pool_feature.h5", "The path of feature file")
flags.DEFINE_string("caption_file_path", "{}.txt", "The path of Caption file")
flags.DEFINE_bool("ifpool", False, "If use the pooling feature.")


class Config(object):
    def __init__(self):
        self.workspace = FLAGS.workspace
        self.feature_path = self.workspace + FLAGS.feature_path
        self.caption = self.workspace + FLAGS.caption_file_path
        self.ifpool = FLAGS.ifpool  # Default feature is fc / if use the pooling feature, it needs to be True

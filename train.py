# @ Author: Dagui Chen/ Xin Su
# @ Email: goblin_chen@163.com/suxin5987@qq.com
# @ Date: 2017-05-08/2017-05-19
# =====================================
from configuration import Config
from utils import CharaterTable, Preprocessor, vetorize_caption
from CaptionModel import CaptionModel
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("caption_len", 30, "The length of caption")
flags.DEFINE_bool("if_finetune", False, "If you need to finetune from the weight file"
                  "Please set this as True and change the weight dir")
flags.DEFINE_string("weight_dir", "/path/to/weight/file.", "The finutune pretrain weight dir")

if_finetune = FLAGS.if_finetune
weight_dir = FLAGS.weight_dir
caption_len = FLAGS.caption_len

config = Config()
data = Preprocessor(config)
ctable = CharaterTable(data.train_captions + data.val_captions)


Y_train = vetorize_caption(data.train_captions, ctable, caption_len)
Y_val = vetorize_caption(data.val_captions, ctable, caption_len)


caption_model = CaptionModel(image_len=data.image_len,
                             caption_len=caption_len,
                             vocab_size=ctable.vocab_size,
                             ifpool=config.ifpool)

if if_finetune:
    caption_model.build_train_model_from_checkpoint(weight_dir)
else:
    caption_model.build_train_model()

caption_model.train(data.train_set, Y_train, data.val_set, Y_val)

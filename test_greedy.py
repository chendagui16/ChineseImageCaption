from configuration import Config
from utils import CharaterTable, Preprocessor
from CaptionModel import CaptionModel
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("caption_len", 25, "The length of caption")
flags.DEFINE_string("model_weights", "/home/suxin/ImageCaption/ImageCaption_coding_test/checkpoint/weights.031-0.776.hdf5", "The weights file of test model")

config = Config()
data = Preprocessor(config)
ctable = CharaterTable(data.train_captions + data.val_captions)

caption_len = FLAGS.caption_len

caption_model = CaptionModel(image_len=data.image_len,
                             caption_len=caption_len,
                             vocab_size=ctable.vocab_size,
                             ifpool=config.ifpool)

caption_model.build_inference_model(FLAGS.model_weights, beam_search=False)
result = caption_model.inference(data.val_set)
num = result.shape[0]

captions = [ctable.decode(result[i], calc_argmax=False) for i in range(num)]

for i, caption in enumerate(captions):
    print i+8001,
    for word in caption:
        print word,
    print ''

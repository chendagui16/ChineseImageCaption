from configuration import Config
from utils import CharaterTable, Preprocessor
from CaptionModel import CaptionModel
from beam_search import CaptionGenerator
import codecs
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("caption_len", 30, "The length of caption")
flags.DEFINE_string("model_weights", "/home/dagui/Documents/class-slides/pattern recognization/course_project/checkpoint/weights.058-0.808.hdf5", "The weights file of test model")  # no use in test
flags.DEFINE_string("save_result", "result.txt", "the file of save the result")

config = Config()
data = Preprocessor(config)
ctable = CharaterTable(data.train_captions + data.val_captions)
caption_len = FLAGS.caption_len

caption_model = CaptionModel(image_len=data.image_len,
                             caption_len=caption_len,
                             vocab_size=ctable.vocab_size,
                             ifpool=config.ifpool)

model_weights = FLAGS.model_weights
caption_model.build_inference_model(model_weights, beam_search=True)
caption_gen = CaptionGenerator(model=caption_model,
                               ctable=ctable,
                               caption_len=caption_len,
                               beam_size=3,  # set beam_search size
                               length_normalization_factor=0.5)  # biger indicate longer sentence will be favored

with codecs.open(FLAGS.save_result, 'w+', 'utf8') as f:
    for id in range(data.test_num):
        # print(id)
        # print(data.test_set[id].shape)
        result = caption_gen.beam_search(data.test_set[id])
        decode = ctable.decode(result[0], calc_argmax=False)
        f.write('{}'.format(9000+id))
        for word in decode:
            f.write(' ' + word)
        if id % 10 == 0:
            print '[%.2f%%]' % ((id*100.0)/data.test_num)
        f.write('\n')

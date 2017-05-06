from configuration import Config
from utils import CharaterTable, Preprocessor, vetorize_caption
from CaptionModel import CaptionModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', default='checkpoint/weights.008-0.866.hdf5', help='checkpoints path')
args = parser.parse_args()

# Load config file, set data location
config = Config()

# preprocess the data
data = Preprocessor(config)

# build the dictionary, convert label to one-hot vector
ctable = CharaterTable(data.train_captions + data.val_captions)
Y_train = vetorize_caption(data.train_captions, ctable, data.caption_len)
Y_val = vetorize_caption(data.val_captions, ctable, data.caption_len)

# Load model config
caption_model = CaptionModel(image_len=data.image_len,
                             caption_len=data.caption_len,
                             vocab_size=ctable.vocab_size)

# build the inference model
caption_model.build_inference_model(args.checkpoint)

# inference the caption
result = caption_model.inference(data.val_set)

# decode the label
num = result.shape[0]
captions = [ctable.decode(result[i], calc_argmax=False) for i in range(num)]

# print the caption, you can write your save code
for i, caption in enumerate(captions):
    print i+8001, caption

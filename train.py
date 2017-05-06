from configuration import Config
from utils import CharaterTable, Preprocessor, vetorize_caption
from CaptionModel import CaptionModel

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

# build train model
caption_model.build_train_model()

# train model
caption_model.train(data.train_set, Y_train, data.val_set, Y_val)

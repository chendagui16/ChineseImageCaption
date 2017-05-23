# @ Author: Dagui Chen/ Xin Su
# @ Email: goblin_chen@163.com/suxin5987@qq.com
# @ Date: 2017-05-08/2017-05-19
# =====================================
import numpy as np
import random
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Embedding, GRU, LSTM, SimpleRNN, RepeatVector, Conv2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("embedding_size", 256, "The size of language word embedding.")
flags.DEFINE_integer("image_embedding_size", 128, "The size of image embedding size")
flags.DEFINE_integer("RNN_out_units", 512, "The number of RNN units.")
flags.DEFINE_integer("batch_size", 20, "Training batch size")
flags.DEFINE_integer("inference_batch_size", 1000, "The batch size of Test")
flags.DEFINE_integer("epochs", 500, "The epochs of Training")
flags.DEFINE_integer("num_RNN_layers", 3, "The Layer number of RNN")
flags.DEFINE_string("save_path", ".", "The path of save dir")
flags.DEFINE_string("RNN_category", "LSTM", "Set the category of RNNCell, GRU, LSTM or SimpleRNN")


class CaptionModel(object):
    def __init__(self, image_len, caption_len, vocab_size, ifpool):
        # image feature length, such as fc2 or fc1
        self.image_len = image_len

        # max caption length
        self.caption_len = caption_len + 2  # add start word and end word

        # unique vocabular size
        self.vocab_size = vocab_size

        self.ifpool = ifpool

        # save_path for checkpoint and tensorboard
        self.save_path = FLAGS.save_path

        # pooling feature shape
        self.pooling_shape = (7, 7, 512)
        self.conv_channel = 512

        # embedding_size (default = 128) word embedding size
        self.embedding_size = FLAGS.embedding_size
        self.image_embedding_size = FLAGS.image_embedding_size
        self.RNN_out_uints = FLAGS.RNN_out_units
        self.batch_size = FLAGS.batch_size
        self.inference_batch_size = FLAGS.inference_batch_size
        self.epochs = FLAGS.epochs
        self.num_RNN_layers = FLAGS.num_RNN_layers
        self.RNN = {'GRU': GRU, 'LSTM': LSTM, 'SimpleRNN': SimpleRNN}[FLAGS.RNN_category]

    def build_train_model(self):
        if self.ifpool:
            image_input = Input(shape=self.pooling_shape, name='image_input')
        else:
            image_input = Input(shape=(self.image_len,), name='image_input')

        caption_input = Input(shape=(self.caption_len,), name='caption_input')

        image_model = Sequential(name='image_model')
        if self.ifpool:
            image_model.add(Conv2D(self.conv_channel, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=self.pooling_shape))
            image_model.add(Conv2D(self.conv_channel, kernel_size=3, strides=1, padding='same', activation='relu'))
            image_model.add(Conv2D(self.conv_channel, kernel_size=3, strides=1, padding='same', activation='relu'))
            image_model.add(GlobalAveragePooling2D())
            image_model.add(RepeatVector(self.caption_len))
            image_model.add(self.RNN(self.image_embedding_size, return_sequences=True))
        else:
            image_model.add(Dense(self.conv_channel, activation='relu', input_shape=(self.image_len,)))
            image_model.add(RepeatVector(self.caption_len))
            image_model.add(self.RNN(self.image_embedding_size, return_sequences=True))

        language_model = Sequential(name='language_model')
        language_model.add(Embedding(self.vocab_size, self.embedding_size, input_length=self.caption_len))

        image_embedding = image_model(image_input)
        caption_embedding = language_model(caption_input)

        RNN_input = Concatenate(axis=-1)([image_embedding, caption_embedding])

        for layer_idx in range(self.num_RNN_layers):
            if layer_idx == 0:
                locals()['RNN_output%s' % layer_idx] = self.RNN(self.RNN_out_uints, name='RNN%s' % layer_idx, return_sequences=True)(RNN_input)
            else:
                locals()['RNN_output%s' % layer_idx] = self.RNN(self.RNN_out_uints, name='RNN%s' % layer_idx, return_sequences=True)(locals()['RNN_output%s' % (layer_idx-1)])

        caption_output = Dense(self.vocab_size, activation='softmax', name='output')(locals()['RNN_output%s' % (self.num_RNN_layers-1)])
        self.model = Model([image_input, caption_input], caption_output)
        self.model.summary()
        self.model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')

    def generator(self, imageData, captionData):
        """Generator for train or val
        Shuffle the training samples
        For each image, we only choose a caption randomly from its 5 captions
        Arg:
            imageData: image feature
            captionData: several one-hot vector for each image
        """
        sampleNum = imageData.shape[0]
        # train set and validation set has different steps_per_epoch, so we need compute again
        steps_per_epoch = sampleNum // self.batch_size
        while 1:
            idx = np.arange(sampleNum)
            np.random.shuffle(idx)
            X1 = np.zeros((self.batch_size, self.image_len))
            X2 = np.zeros((self.batch_size, self.caption_len))
            Y = np.zeros((self.batch_size, self.caption_len, self.vocab_size))
            Y_end = np.zeros((self.batch_size, 1, self.vocab_size))
            Y_end[:, 0, 0] = 1  # The end flag for caption
            for i in range(steps_per_epoch):
                randidx = idx[i*self.batch_size:(i+1)*self.batch_size]
                X1 = imageData[randidx]
                Y = np.array(map(random.choice, [captionData[j] for j in randidx]))
                X2 = np.argmax(Y, axis=-1)
                yield ({'image_input': X1, 'caption_input': X2}, {'output': np.concatenate([Y[:, 1:], Y_end], axis=1)})

    def build_train_model_from_checkpoint(self, checkpoint):
        self.model = load_model(checkpoint)

    def train(self, X_train, Y_train, X_val, Y_val):
        train_num = len(Y_train)
        val_num = len(Y_val)
        steps_per_epoch = train_num // self.batch_size
        val_steps = val_num // self.batch_size
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001)
        save_model = ModelCheckpoint(self.save_path+'/checkpoint/weights.{epoch:03d}-{val_acc:.3f}.hdf5', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.save_path+'/tf_logs')
        self.model.fit_generator(self.generator(X_train, Y_train),
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=self.generator(X_val, Y_val),
                                 validation_steps=val_steps,
                                 callbacks=[reduce_lr, save_model, tensorboard],
                                 epochs=self.epochs)

    def build_inference_model(self, checkpoint, beam_search=False):
        if beam_search:
            self.inference_batch_size = 1
            model = load_model(checkpoint)

        # image model
        self.image_model = Sequential()
        if self.ifpool:
            self.image_model.add(Conv2D(self.conv_channel, trainable=False, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=self.pooling_shape))
            self.image_model.add(Conv2D(self.conv_channel, trainable=False, kernel_size=3, strides=1, padding='same', activation='relu'))
            self.image_model.add(Conv2D(self.conv_channel, trainable=False, kernel_size=3, strides=1, padding='same', activation='relu'))
            self.image_model.add(GlobalAveragePooling2D())
            self.image_model.add(RepeatVector(self.caption_len))
            self.image_model.add(self.RNN(self.image_embedding_size, return_sequences=True, trainable=False))
        else:
            self.image_model.add(Dense(self.conv_channel, activation='relu', trainable=False, input_shape=(self.image_len,)))
            self.image_model.add(RepeatVector(self.caption_len))
            self.image_model.add(self.RNN(self.image_embedding_size, return_sequences=True, trainable=False))

        # copy weights from loaded model
        image_layer = model.get_layer('sequential_1')
        assert image_layer is not None, 'There is no layer named sequential_1'
        self.image_model.set_weights(image_layer.get_weights())

        # language model
        self.language_model = Sequential()
        self.language_model.add(Embedding(self.vocab_size, self.embedding_size,
                                          trainable=False, input_length=1))  # set input_length=1 so as to forward one-step
        # copy weights from loaded model
        language_layer = model.get_layer('sequential_2')
        self.language_model.set_weights(language_layer.get_weights())

        # caption model
        # Forward a image/word embedding to get a next word
        # Use Stateful RNN
        self.caption_model = Sequential()
        for idx in range(self.num_RNN_layers):
            if idx == 0:
                self.caption_model.add(self.RNN(self.RNN_out_uints, return_sequences=True, trainable=False, stateful=True,
                                                batch_input_shape=(self.inference_batch_size, 1, self.embedding_size + self.image_embedding_size)))
            else:
                self.caption_model.add(self.RNN(self.RNN_out_uints, return_sequences=True, trainable=False, stateful=True))

        self.caption_model.add(Dense(self.vocab_size, activation='softmax', trainable=False))

        caption_weigts = []
        for layer in model.layers[-(self.num_RNN_layers+1):]:  # copy the last (num_rnn_layers+1) layer's weights
            caption_weigts.extend(layer.get_weights())

        self.caption_model.set_weights(caption_weigts)
        del model

    def get_image_output(self, test_batch):
        return self.image_model.predict_on_batch(test_batch)

    def inference(self, X_test):
        """Inference using greedy method
        """
        test_num = X_test.shape[0]
        assert test_num % self.inference_batch_size == 0, 'inference_batch_size should divide Test sample number'
        steps_per_epoch = test_num // self.inference_batch_size
        result = np.zeros((0, self.caption_len))
        for i in range(steps_per_epoch):
            test_batch = X_test[i*self.inference_batch_size:(i+1)*self.inference_batch_size]
            char = np.zeros((self.inference_batch_size, self.caption_len))
            image_output = self.get_image_output(test_batch)

            self.caption_model.reset_states()
            predict = self.caption_model.predict_on_batch(image_output)

            for j in range(self.caption_len):
                char[:, j] = np.argmax(predict, axis=-1).squeeze()
                language_output = self.language_model.predict_on_batch(char[:, j])
                predict = self.caption_model.predict_on_batch(language_output)

            result = np.concatenate([result, char], axis=0)

        return result

    def inference_step(self, image_embedding, sentence_feed):
        """Used for Beam search
        Get the next predict word and prob
        Args:
            image_embedding: the image_model's output
            part_caption: the index to part caption word
        """
        self.caption_model.reset_states()
        for step, word_idx in enumerate(sentence_feed):
            word = np.array([word_idx])
            language_output = self.language_model.predict_on_batch(word[None, ...])
            caption_input = np.concatenate([image_embedding[0, step][None, None, ...], language_output], axis=-1)
            next_predict = self.caption_model.predict_on_batch(caption_input)
        return next_predict

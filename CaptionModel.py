import numpy as np
import random
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Embedding, GRU, RepeatVector
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


class CaptionModel(object):
    def __init__(self, image_len, caption_len, vocab_size, save_path='.'):
        # image feature length, such as fc2 or fc1
        self.image_len = image_len

        # max caption length
        self.caption_len = caption_len

        # unique vocabular size
        self.vocab_size = vocab_size

        # save_path for checkpoint and tensorboard
        self.save_path = save_path

        # embedding_size (default = 128) word embedding size
        self.embedding_size = 128
        self.RNN_out_uints = 128
        self.batch_size = 40
        self.inference_batch_size = 1000  # inference_batch_size should divide Test sample number
        self.epochs = 40

    def build_train_model(self):
        image_input = Input(shape=(self.image_len,), name='image_input')
        caption_input = Input(shape=(self.caption_len,), name='caption_input')

        image_model = Sequential(name='image_model')
        image_model.add(Dense(self.embedding_size, activation='relu', input_shape=(self.image_len,)))
        image_model.add(RepeatVector(1))

        language_model = Sequential(name='language_model')
        language_model.add(Embedding(self.vocab_size, self.embedding_size, input_length=self.caption_len))

        image_embedding = image_model(image_input)
        caption_embedding = language_model(caption_input)

        RNN_input = Concatenate(axis=-2)([image_embedding, caption_embedding])
        RNN_output = GRU(self.RNN_out_uints, name='RNN', return_sequences=True)(RNN_input)
        caption_output = Dense(self.vocab_size, activation='softmax', name='output')(RNN_output)

        self.model = Model([image_input, caption_input], caption_output)
        self.model.summary()
        self.model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')

    def generator(self, imageData, captionData):
        """Generator for train or val
        Shuffle the training samples
        For each image, we only choose a caption randomly from its 5 captions
        @ imageData: image feature
        @ captionData: several one-hot vector for each image
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
                yield ({'image_input': X1, 'caption_input': X2}, {'output': np.concatenate([Y, Y_end], axis=1)})

    def train(self, X_train, Y_train, X_val, Y_val):
        train_num = len(Y_train)
        val_num = len(Y_val)
        steps_per_epoch = train_num // self.batch_size
        val_steps = val_num // self.batch_size
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)
        save_model = ModelCheckpoint(self.save_path+'/checkpoint/weights.{epoch:03d}-{val_acc:.3f}.hdf5', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.save_path+'/logs')
        self.model.fit_generator(self.generator(X_train, Y_train),
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=self.generator(X_val, Y_val),
                                 validation_steps=val_steps,
                                 callbacks=[reduce_lr, save_model, tensorboard],
                                 epochs=self.epochs)

    def build_inference_model(self, checkpoint):
        try:
            model = load_model(checkpoint)
        except:
            print('No checkpoint in the {}'.format(checkpoint))

        # image model
        self.image_model = Sequential()
        self.image_model.add(Dense(self.embedding_size, activation='relu', trainable=False, input_shape=(self.image_len,)))
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
        self.caption_model.add(GRU(self.RNN_out_uints, return_sequences=True, trainable=False,
                                   stateful=True, batch_input_shape=(self.inference_batch_size, 1, self.embedding_size)))
        self.caption_model.add(Dense(self.vocab_size, activation='softmax', trainable=False))

        caption_weigts = []
        for layer in model.layers[-2:]:  # copy the last 2 layer's weights
            caption_weigts.extend(layer.get_weights())

        self.caption_model.set_weights(caption_weigts)
        del model

    def inference(self, X_test):
        test_num = X_test.shape[0]
        assert test_num % self.inference_batch_size == 0, 'inference_batch_size should divide Test sample number'
        steps_per_epoch = test_num // self.inference_batch_size
        result = np.zeros((0, self.caption_len))
        for i in range(steps_per_epoch):
            test_batch = X_test[i*self.inference_batch_size:(i+1)*self.inference_batch_size]
            char = np.zeros((self.inference_batch_size, self.caption_len))
            image_output = np.expand_dims(self.image_model.predict_on_batch(test_batch), axis=1)

            self.caption_model.reset_states()
            predict = self.caption_model.predict_on_batch(image_output)

            for j in range(self.caption_len):
                char[:, j] = np.argmax(predict, axis=-1).squeeze()
                language_output = self.language_model.predict_on_batch(char[:, j])
                predict = self.caption_model.predict_on_batch(language_output)

            result = np.concatenate([result, char], axis=0)

        return result

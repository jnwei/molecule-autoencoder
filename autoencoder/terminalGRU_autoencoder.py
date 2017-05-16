import warnings
import numpy as np
from random import shuffle
import time

from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint

from variationaldense import VariationalDense as VAE
from terminalgru import TerminalGRU
from vaeweightannealer import VAEWeightAnnealer
import hyperparams


MAX_LEN = 120
TRAIN_SET = 'drugs'
TEMPERATURE = np.array(1.00, dtype=np.float32)
PADDING = 'right'

CONFIGS = {
    'drugs': {'file': '../data/250k_rndm_zinc_drugs_clean.smi',
              'chars': [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
                        '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
                        'c', 'l', 'n', 'o', 'r', 's']},
}

CHARS = CONFIGS[TRAIN_SET]['chars']
NCHARS = len(CHARS)
CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))


def smile_convert(string):
    if len(string) < MAX_LEN:
        if PADDING == 'right':
            return string + " " * (MAX_LEN - len(string))
        elif PADDING == 'left':
            return " " * (MAX_LEN - len(string)) + string
        elif PADDING == 'none':
            return string


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class CheckMolecule(Callback):
    def on_epoch_end(self, epoch, logs={}):
        test_smiles = ["c1ccccc1"]
        test_smiles = [smile_convert(i) for i in test_smiles]
        Z = np.zeros((len(test_smiles), MAX_LEN, NCHARS), dtype=np.bool)
        for i, smile in enumerate(test_smiles):
            for t, char in enumerate(smile):
                Z[i, t, CHAR_INDICES[char]] = 1

        string = ""
        for i in self.model.predict(Z):
            for j in i:
                index = sample(j, TEMPERATURE)
                string += INDICES_CHAR[index]
        print("\n" + string)


class CheckpointPostAnnealing(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', start_epoch=0):
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose,
                                 save_best_only=save_best_only, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.start_epoch:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath)))
                        self.best = current
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print(('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor)))
            else:
                if self.verbose > 0:
                    print(('Epoch %05d: saving model to %s' % (epoch, filepath)))
                self.model.save_weights(filepath, overwrite=True)


def main(training_path,
         parameters,
         weight_file='weights.h5',
         model_file='model.json',
         limit=None):
    for key in parameters:
        if type(parameters[key]) in [float, np.ndarray]:
            parameters[key] = np.float(parameters[key])
        print(key, parameters[key])

    def no_schedule(x):
        return float(1)

    def sigmoid_schedule(x, slope=1., start=parameters['vae_annealer_start']):
        return float(1 / (1. + np.exp(slope * (start - float(x)))))

    start = time.time()
    with open(training_path, 'r') as f:
        smiles = f.readlines()
    smiles = [i.strip() for i in smiles]
    if limit is not None:
        smiles = smiles[:limit]
    print('Training set size is', len(smiles))
    smiles = [smile_convert(i) for i in smiles if smile_convert(i)]
    print('Training set size is {}, after filtering to max length of {}'.format(len(smiles), MAX_LEN))
    shuffle(smiles)

    print(('total chars:', NCHARS))

    X = np.zeros((len(smiles), MAX_LEN, NCHARS), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            X[i, t, CHAR_INDICES[char]] = 1

    model = Sequential()

    ## Convolutions
    model.add(Convolution1D(int(parameters['conv_dim_depth'] *
                                parameters['conv_d_growth_factor']),
                            int(parameters['conv_dim_width'] *
                                parameters['conv_w_growth_factor']),
                            batch_input_shape=(parameters['batch_size'], MAX_LEN, NCHARS),
                            activation=parameters['conv_activation']))

    ## Middle layers
    model.add(Dense(int(parameters['hidden_dim'] *
                            parameters['hg_growth_factor']**(parameters['middle_layer'] - i)),
                        activation=parameters['activation']))
    model.add(BatchNormalization(mode=0, axis=-1))

    #model.add(GRU(parameters['recurrent_dim'], return_sequences=True,
    #                  activation=parameters['rnn_activation']))
    #model.add(BatchNormalization(mode=0, axis=-1))

    model.add(TerminalGRU(NCHARS, 
                              return_sequences=True,
                              activation='softmax',
                              temperature=TEMPERATURE,
                              dropout_U=parameters['tgru_dropout']))

    if parameters['optim'] == 'adam':
        optim = Adam(lr=parameters['lr'], beta_1=parameters['momentum'])
    elif parameters['optim'] == 'rmsprop':
        optim = RMSprop(lr=parameters['lr'], beta_1=parameters['momentum'])
    elif parameters['optim'] == 'sgd':
        optim = SGD(lr=parameters['lr'], beta_1=parameters['momentum'])

    model.compile(loss=parameters['loss'], optimizer=optim)

    # SAVE

    json_string = model.to_json()
    open(model_file, 'w').write(json_string)

    print(parameters)

    # CALLBACK
    smile_checker = CheckMolecule()

    cbk = ModelCheckpoint(weight_file,
                          save_best_only=True)

    model.fit(X, X, batch_size=parameters['batch_size'],
                  nb_epoch=parameters['epochs'],
                  callbacks=[smile_checker, cbk],
                  validation_split=parameters['val_split'],
                  show_accuracy=True)

    end = time.time()
    print(parameters)
    print((end - start), 'seconds elapsed')


if __name__ == "__main__":
    main(training_path=CONFIGS[TRAIN_SET]['file'],
         parameters=hyperparams.simple_params(),
         limit=5000) # just train on first 5000 molecules for quick testing.  set to None to use all 250k

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf

import tuiautopilotml.base_helpers as h
from tuiautopilotml.configs import scoring_metrics


def y_train_handler(y_train, activation_f_type):
    if activation_f_type in ('classif', 'multiclass'):
        return tf.keras.utils.to_categorical(y_train)
    else:
        return y_train


def cv_eval_mlp(df, target_label, activation_f_type='classif', optimizer='adam', regulator=20,
                hl_activation='relu', evaluation_metric='accuracy', metric_to_monitor='loss', mode='min',
                epochs=100, batch_size=128, n_folds=10, n_repeats=3, patience=10, verbose=1, random_state=0,
                scaler=None):
    """Cross validation for MLP"""

    # Observe that x and y should be scaled
    x, y = h.get_x_y_from_df(df, target_label, scaled_df=True, scaler=scaler)
    results = list()

    # define evaluation procedure
    cv_folds = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    early_stop = EarlyStopping(monitor=metric_to_monitor, mode=mode, verbose=verbose, patience=patience)
    model = None
    score = None
    for train_ix, test_ix in cv_folds.split(x):
        x_train, x_test = x[train_ix], x[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        model = compile_mlp(x, y, activation_f_type=activation_f_type, optimizer=optimizer,
                            regulator=regulator, hl_activation=hl_activation, evaluation_metric=evaluation_metric)

        y_train = y_train_handler(y_train=y_train, activation_f_type=activation_f_type)

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, callbacks=[early_stop], batch_size=batch_size)
        # evaluate model on test set
        prediction = model.predict(x_test, batch_size=batch_size, verbose=verbose)
        prediction = np.argmax(prediction, axis=1)

        if activation_f_type == 'classif':
            score = scoring_metrics['clf'][evaluation_metric](y_test, prediction)
        elif activation_f_type == 'reg':
            score = scoring_metrics['reg'][evaluation_metric](y_test, prediction)

        results.append(score)

    results = {'mlp_model': [np.mean(results), np.std(results)]}

    return results, model


"""******** KERAS-MLP MODEL ******** """


def get_mlp_initial_params(activation_f_type: str):
    if activation_f_type == 'classif':
        o_activation = 'sigmoid'
        loss = 'binary_crossentropy'
        return o_activation, loss

    elif activation_f_type == 'multiclass':
        o_activation = 'softmax'
        loss = 'categorical_crossentropy'
        return o_activation, loss

    elif activation_f_type == 'reg':
        o_activation = 'linear'
        loss = 'mean_squared_error'
        return o_activation, loss


def compile_mlp(x, y, activation_f_type='classif', optimizer='adam', regulator=20, hl_activation='relu',
                evaluation_metric='accuracy'):
    """
    Info: MLP for baseline model creation. Regulator helps you to increase or decrease n_neurons

    Args:
        x:
        y:
        activation_f_type:
        optimizer:
        regulator:
        hl_activation:
        evaluation_metric:

    Returns:

    """

    n_inputs = x.shape[1]
    n_outputs = int(y.nunique())

    o_activation, loss = get_mlp_initial_params(activation_f_type=activation_f_type)

    n_neurons = int(np.sqrt(n_inputs * n_outputs) * regulator)

    print(f'Number of neurons: {n_neurons}-{int(n_neurons / 2.5)}-{int(n_neurons / 5.5)}')

    model = Sequential()

    model.add(Dense(n_neurons, input_dim=n_inputs, activation=hl_activation))
    model.add(Dropout(0.3))

    model.add(Dense(int(n_neurons / 2.5), activation=hl_activation))
    model.add(Dropout(0.3))

    model.add(Dense(int(n_neurons / 5.5), activation=hl_activation))
    model.add(Dropout(0.1))

    model.add(Dense(n_outputs, activation=o_activation))  # output layer

    model.compile(loss=loss, optimizer=optimizer, metrics=[evaluation_metric])

    return model

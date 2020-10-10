from tensorflow import keras


def get_optimizer(config: dict):
    optimizer = None
    print(config)
    type = config.pop('type', 'sgd')
    learning_rate = config.pop('learning_rate', 1e-3)
    if type == 'sgd':
        momentum = config.pop('momentum', 0.9)
        nesterov = config.pop('nesterov', True)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate,
                                         momentum=momentum,
                                         nesterov=nesterov)
    elif type == 'adam':
        beta_1 = config.pop('beta_1', 0.9)
        beta_2 = config.pop('beta_2', 0.999)
        amsgrad = config.pop('amsgrad', False)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                          beta_1=beta_1,
                                          beta_2=beta_2,
                                          amsgrad=amsgrad)
    return optimizer, learning_rate

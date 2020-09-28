from tensorflow import keras


def get_optimizer(config: dict, schedule=False):
    optimizer = None
    type = config.pop('type', 'sgd')
    learning_rate = config.get('learning_rate', 0.001)
    if type == 'sgd':
        momentum = config.pop('momentum', 0.9)
        nesterov = config.pop('nesterov', False)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate,
                                         momentum=momentum,
                                         nesterov=nesterov)
        print(f'lr is {learning_rate}')
    elif type == 'adam':
        beta_1 = config.pop('beta_1', 0.9)
        beta_2 = config.pop('beta_2', 0.999)
        amsgrad = config.pop('amsgrad', False)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate,
                                          beta_1=beta_1,
                                          beta_2=beta_2,
                                          amsgrad=amsgrad)
    return optimizer, learning_rate

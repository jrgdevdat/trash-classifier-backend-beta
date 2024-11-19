import numpy as np

class HyperparametersConfiguration:

    def __init__(self, learning_rate, hidden_units, decoder_layers, encoder_trainable):
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.decoder_layers = decoder_layers
        self.encoder_trainable = encoder_trainable

    def __str__(self):
        header = '**** Hyperparameters configuration\n'
        learning_rate_line = 'Learning rate: {}\n'.format(self.learning_rate)
        hidden_units_line = 'Hidden units: {}\n'.format(self.hidden_units)
        decoder_layers_line = 'Decoder layers: {}\n'.format(self.decoder_layers)
        encoder_trainable_line = 'Encoder trainable: {}\n'.format(self.encoder_trainable)
        return '{}{}{}{}{}'.format(
            header, learning_rate_line, hidden_units_line, decoder_layers_line, encoder_trainable_line)

    @classmethod
    def fromRandomness(cls, tuning_hyperparameters_conditions):
        # Obtener una learning rate aleatoria
        exp_inf = tuning_hyperparameters_conditions.min_learning_rate_exponent
        exp_sup = tuning_hyperparameters_conditions.max_learning_rate_exponent
        learning_rate = 10 ** (exp_inf + (exp_sup - exp_inf) * np.random.rand())
        # Obtener un valor para hidden units aleatorio
        min_hidden_units = tuning_hyperparameters_conditions.min_hidden_units
        max_hidden_units = tuning_hyperparameters_conditions.max_hidden_units
        hidden_units = np.random.randint(min_hidden_units, max_hidden_units + 1)
        # Obtener un valor para decoder layers
        min_decoder_layers = tuning_hyperparameters_conditions.min_decoder_layers
        max_decoder_layers = tuning_hyperparameters_conditions.max_decoder_layers
        decoder_layers = np.random.randint(min_decoder_layers, max_decoder_layers + 1)
        # Switch para entrenar o no el encoder
        encoder_trainable = tuning_hyperparameters_conditions.encoder_trainable
        return cls(learning_rate, hidden_units, decoder_layers, encoder_trainable)

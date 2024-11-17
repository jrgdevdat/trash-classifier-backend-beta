
class HyperparametersConfiguration:

    def __init__(self, learning_rate, hidden_units, decoder_layers):
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.decoder_layers = decoder_layers

    def __str__(self):
        header = '**** Hyperparameters configuration\n'
        learning_rate_line = 'Learning rate: {}\n'.format(self.learning_rate)
        hidden_units_line = 'Hidden units: {}\n'.format(self.hidden_units)
        decoder_layers_line = 'Decoder layers: {}\n'.format(self.decoder_layers)
        return '{}{}{}{}'.format(header, learning_rate_line, hidden_units_line, decoder_layers_line)
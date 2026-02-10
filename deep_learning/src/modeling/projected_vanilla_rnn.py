from torch.nn import Module
from torch.nn.functional import relu
from torch import tanh
from .modules import PretrainedEmbeddingLayer, CellLayer, MLP, SequentialModel, LastState, AvgPoolingState


class ProjectedVanillaRNN(Module):
    def __init__(self, embeddings, avg_pooling=False,
                 embeddings_dropout=0.3,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=2,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):

        super(ProjectedVanillaRNN, self).__init__()
        self.input_list = ['text']
        self.name = "ProjectedVanillaRNN_AVG" if avg_pooling else "ProjectedVanillaRNN_LAST"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, dropout=embeddings_dropout, trainable=False)
        self.projection_layer = MLP(num_of_layers=1, init_size=self.word_embedding_layer.get_output_size(),
                                    out_size=128, outer_activation=tanh)
        self.cell = CellLayer(is_gru, self.projection_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=large_size,
                                  out_size=4,
                                  dropout=top_mlp_dropout,
                                  inner_activation=top_mlp_activation,
                                  outer_activation=top_mlp_outer_activation)

        if avg_pooling:
            self.state = AvgPoolingState(large_size, large_size)
        else:
            self.state = LastState(large_size, large_size)
        self.seq = SequentialModel([self.word_embedding_layer, self.projection_layer, self.cell,
                                    self.state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out

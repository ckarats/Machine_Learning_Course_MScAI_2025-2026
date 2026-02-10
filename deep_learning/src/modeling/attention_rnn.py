from torch.nn import Module
from torch.nn.functional import relu
from .modules import PretrainedEmbeddingLayer, CellLayer, MLP, AttendedState, SequentialModel


class AttentionRNN(Module):
    def __init__(self, embeddings,
                 trainable_embeddings=True,
                 embeddings_dropout=0.3,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.0,
                 top_mlp_layers=2,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        super(AttentionRNN, self).__init__()
        self.input_list = ['text']
        self.name = "AttentionRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=large_size,
                                  out_size=4,
                                  dropout=top_mlp_dropout,
                                  inner_activation=top_mlp_activation,
                                  outer_activation=top_mlp_outer_activation)
        self.state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out
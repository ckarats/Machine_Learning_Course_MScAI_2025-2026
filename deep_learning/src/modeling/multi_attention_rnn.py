from torch.nn import Module
import torch
from torch.nn.functional import relu
from .modules import PretrainedEmbeddingLayer, CellLayer, MLP, AttendedState, SequentialModel, ConcatenationLayer


class MultiAttentionRNN(Module):
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

        super(MultiAttentionRNN, self).__init__()
        self.input_list = ['text']
        self.name = "MultiAttentionRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer_harassment = MLP(num_of_layers=top_mlp_layers,
                                             init_size=large_size,
                                             out_size=1,
                                             dropout=top_mlp_dropout,
                                             inner_activation=top_mlp_activation,
                                             outer_activation=top_mlp_outer_activation)
        self.decision_layer_sexual = MLP(num_of_layers=top_mlp_layers,
                                         init_size=large_size,
                                         out_size=1,
                                         dropout=top_mlp_dropout,
                                         inner_activation=top_mlp_activation,
                                         outer_activation=top_mlp_outer_activation)
        self.decision_layer_physical = MLP(num_of_layers=top_mlp_layers,
                                           init_size=large_size,
                                           out_size=1,
                                           dropout=top_mlp_dropout,
                                           inner_activation=top_mlp_activation,
                                           outer_activation=top_mlp_outer_activation)
        self.decision_layer_indirect = MLP(num_of_layers=top_mlp_layers,
                                           init_size=large_size,
                                           out_size=1,
                                           dropout=top_mlp_dropout,
                                           inner_activation=top_mlp_activation,
                                           outer_activation=top_mlp_outer_activation)
        self.last_state_harassment = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.last_state_indirect = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.last_state_sexual = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.last_state_physical = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        representation = self.seq(x)
        harasment_attention = self.last_state_harassment(representation)
        indirect_attention = self.last_state_indirect(representation)
        sexual_attention = self.last_state_sexual(representation)
        physical_attention = self.last_state_physical(representation)
        out_harassment = self.decision_layer_harassment(harasment_attention)
        out_indirect = self.decision_layer_indirect(indirect_attention)
        out_sexual = self.decision_layer_sexual(sexual_attention)
        out_physical = self.decision_layer_physical(physical_attention)
        out = torch.cat([out_harassment, out_sexual, out_physical, out_indirect], dim=1)
        return out

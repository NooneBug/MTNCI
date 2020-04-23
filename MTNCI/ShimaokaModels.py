import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class CharEncoder(nn.Module):
    def __init__(self, char_vocab, args):
        super(CharEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conv_dim_input = 100
        filters = 5
        self.char_W = nn.Embedding(char_vocab.size(), conv_dim_input, padding_idx=0)
        self.conv1d = nn.Conv1d(conv_dim_input, args.char_emb_size, filters)  # input, output, filter_number

    def forward(self, span_chars):
        char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
        conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
        conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
        cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
        cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
        return cnn_output
class MentionEncoder(nn.Module):

    def __init__(self, char_vocab, args):
        super(MentionEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.char_encoder = CharEncoder(char_vocab, args)
        self.attentive_weighted_average = SelfAttentiveSum(args.emb_size, 1)
        self.dropout = nn.Dropout(args.mention_dropout)

    def forward(self, mentions, mention_chars, word_lut):
        mention_embeds = word_lut(mentions)             # batch x mention_length x emb_size

        weighted_avg_mentions, _ = self.attentive_weighted_average(mention_embeds)
        char_embed = self.char_encoder(mention_chars)
        output = torch.cat((weighted_avg_mentions, char_embed), 1)
        return self.dropout(output).cuda()


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.emb_size = args.emb_size
        self.pos_emb_size = args.positional_emb_size
        self.rnn_size = args.context_rnn_size
        self.hidden_attention_size = 100
        super(ContextEncoder, self).__init__()
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.context_dropout = nn.Dropout(args.context_dropout)
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
        self.attention = SelfAttentiveSum(self.rnn_size * 2, self.hidden_attention_size) # x2 because of bidirectional

    def forward(self, contexts, positions, context_len, word_lut, hidden=None):
        """
        :param contexts: batch x max_seq_len
        :param positions: batch x max_seq_len
        :param context_len: batch x 1
        """
        positional_embeds = self.get_positional_embeddings(positions)   # batch x max_seq_len x pos_emb_size
        ctx_word_embeds = word_lut(contexts)                            # batch x max_seq_len x emb_size
        ctx_embeds = torch.cat((ctx_word_embeds, positional_embeds), 2)

        ctx_embeds = self.context_dropout(ctx_embeds)

        rnn_output = self.sorted_rnn(ctx_embeds, context_len)

        return self.attention(rnn_output)

    def get_positional_embeddings(self, positions):
        """ :param positions: batch x max_seq_len"""
        pos_embeds = self.pos_linear(positions.view(-1, 1))                     # batch * max_seq_len x pos_emb_size
        return pos_embeds.view(positions.size(0), positions.size(1), -1)        # batch x max_seq_len x pos_emb_size

    def sorted_rnn(self, ctx_embeds, context_len):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = self.sort_batch_by_length(ctx_embeds, context_len)
        packed_sequence_input = pack(sorted_inputs, sorted_sequence_lengths, batch_first=True)
        self.rnn.flatten_parameters()
        packed_sequence_output, _ = self.rnn(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = unpack(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def sort_batch_by_length(self, tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
        """
        @ from allennlp
        Sort a batch first tensor by some specified lengths.

        Parameters
        ----------
        tensor : Variable(torch.FloatTensor), required.
            A batch first Pytorch tensor.
        sequence_lengths : Variable(torch.LongTensor), required.
            A tensor representing the lengths of some dimension of the tensor which
            we want to sort by.

        Returns
        -------
        sorted_tensor : Variable(torch.FloatTensor)
            The original tensor sorted along the batch dimension with respect to sequence_lengths.
        sorted_sequence_lengths : Variable(torch.LongTensor)
            The original sequence_lengths sorted by decreasing size.
        restoration_indices : Variable(torch.LongTensor)
            Indices into the sorted_tensor such that
            ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
        """

        if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
            raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

        sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
        sorted_tensor = tensor.index_select(0, permutation_index)
        # This is ugly, but required - we are creating a new variable at runtime, so we
        # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
        # refilling one of the inputs to the function.
        index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
        # This is the equivalent of zipping with index, sorting by the original
        # sequence lengths and returning the now sorted indices.
        index_range = Variable(index_range.long())
        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return sorted_tensor, sorted_sequence_lengths, restoration_indices


class SelfAttentiveSum(nn.Module):
    """
    Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
    """
    def __init__(self, embed_dim, hidden_dim):
        """
        :param embed_dim: in forward(input_embed), the size will be batch x seq_len x emb_dim
        :param hidden_dim:
        """
        super(SelfAttentiveSum, self).__init__()
        self.key_maker = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key_rel = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.key_output = nn.Linear(hidden_dim, 1, bias=False)
        self.key_softmax = nn.Softmax(dim=1)

    def forward(self, input_embed):     # batch x seq_len x emb_dim
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])  # batch * seq_len x emb_dim
        k_d = self.key_maker(input_embed_squeezed)      # batch * seq_len x hidden_dim
        k_d = self.key_rel(k_d)
        if self.hidden_dim == 1:
            k = k_d.view(input_embed.size()[0], -1)     # batch x seq_len
        else:
            k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
        weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)  # batch x seq_len x 1
        weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, embed_dim
        return weighted_values, weighted_keys
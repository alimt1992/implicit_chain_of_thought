import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear', leaky_param=0.2):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        if w_init == 'linear':
            nn.init.xavier_uniform_(
                self.linear_layer.weight,
                gain=nn.init.calculate_gain(w_init))
        else:
            nn.init.xavier_uniform_(
                self.linear_layer.weight,
                gain=nn.init.calculate_gain(w_init, leaky_param))

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, input_dim, num_hidden, num_latent, cross_attn=True, transformer=True,
                 t_nhead=8, t_num_lyrs=6, t_dim_feedforward=2048, t_dropout=0.1):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim * 2, num_hidden)
        self.transformer = transformer
        self.cross_attentions = None
        self.context_projection = None
        self.target_projection = None

        if transformer:
            self.self_attentions = nn.Transformer(d_model=num_hidden, nhead=t_nhead, num_encoder_layers=t_num_lyrs,
                                                  num_decoder_layers=0, dim_feedforward=t_dim_feedforward, activation=nn.GELU(),
                                                  dropout=t_dropout, batch_first=True, norm_first=True)

        else:
            self.self_attentions = nn.ModuleList([Attention(num_hidden, t_nhead) for _ in range(t_num_lyrs)])

        if cross_attn:
            self.cross_attentions = nn.ModuleList([Attention(num_hidden, t_nhead) for _ in range(t_num_lyrs)])
            self.context_projection = Linear(input_dim, num_hidden)
            self.target_projection = Linear(input_dim, num_hidden)

        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='leaky_relu')
        self.mu = Linear(num_hidden, num_latent, w_init='leaky_relu')
        self.log_sigma = Linear(num_hidden, num_latent, w_init='leaky_relu')

    def forward(self, x, y, target_x):
        encoder_input = torch.cat([x,y], dim=-1)
        encoder_input = self.input_projection(encoder_input)

        if self.transformer:
            encoder_input = self.self_attentions(encoder_input, encoder_input)
        else:
            for attention in self.self_attentions:
                encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        hidden = None
        if self.cross_attentions==None:
            hidden = encoder_input.mean(dim=1)
        else:
            query = self.target_projection(target_x)
            keys = self.context_projection(x)

            for attention in self.cross_attentions:
                query, attn = attention(keys, encoder_input, query)
            hidden = query

        hidden = nn.functional.gelu(self.penultimate_layer(hidden))

        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return mu, log_sigma, z

class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """
    def __init__(self, input_dim, num_hidden, transformer=True,
                 t_nhead=8, t_num_lyrs=6, t_dim_feedforward=2048, t_dropout=0.1):
        super(DeterministicEncoder, self).__init__()
        self.transformer = transformer
        if transformer:
            self.self_attentions = nn.Transformer(d_model=num_hidden, nhead=t_nhead, num_encoder_layers=t_num_lyrs,
                                                  num_decoder_layers=0, dim_feedforward=t_dim_feedforward, activation=nn.GELU(),
                                                  dropout=t_dropout, batch_first=True, norm_first=True)

        else:
            self.self_attentions = nn.ModuleList([Attention(num_hidden, t_nhead) for _ in range(t_num_lyrs)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden, t_nhead) for _ in range(t_num_lyrs)])
        self.input_projection = Linear(input_dim * 2, num_hidden)
        self.context_projection = Linear(input_dim, num_hidden)
        self.target_projection = Linear(input_dim, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat([context_x,context_y], dim=-1)

        encoder_input = self.input_projection(encoder_input)

        if self.transformer:
            encoder_input = self.self_attentions(encoder_input, encoder_input)
        else:
            for attention in self.self_attentions:
                encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

class Decoder(nn.Module):
    """
    Decoder for generation
    """
    def __init__(self, output_dim, num_hidden, num_lyrs=6):
        super(Decoder, self).__init__()
        self.target_projection = Linear(output_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='leaky_relu') for _ in range(num_lyrs)])
        self.final_projection = Linear(num_hidden * 3, output_dim)

    def forward(self, r, z, target_x):
        # batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)

        hidden = torch.stack([r, z, target_x], dim=-1)

        for linear in self.linears:
            hidden = nn.functional.gelu(linear(hidden))

        y_pred = self.final_projection(hidden)

        return y_pred

class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        result, attns = self.multihead(key, value, query)

        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        attns = attns.view(self.h, batch_size, seq_q, seq_k)
        attns = attns.permute(1, 0, 2, 3).contiguous().view(batch_size, -1, seq_q, seq_k)

        result = torch.cat([residual, result], dim=-1)

        result = self.final_linear(result)

        result = self.residual_dropout(result)
        result = result + residual

        result = self.layer_norm(result)

        return result, attns


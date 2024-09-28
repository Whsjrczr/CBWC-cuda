from torch import nn
from layers import LayerNorm, MultiHeadAttention, PositionwiseFeedForward, CenteringMatrix


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, scaleonly=False, columncenter=False,
                 dropout=True, rescenter=False):
        super(DecoderLayer, self).__init__()
        self.rescenter = rescenter
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, columncenter=columncenter)
        self.norm1 = LayerNorm(d_model=d_model, scaleonly=scaleonly)
        self.dropout1 = nn.Dropout(p=drop_prob) if dropout else nn.Identity()
        self.rescenter1 = CenteringMatrix()

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, columncenter=columncenter)
        self.norm2 = LayerNorm(d_model=d_model, scaleonly=scaleonly)
        self.dropout2 = nn.Dropout(p=drop_prob) if dropout else nn.Identity()
        self.rescenter2 = CenteringMatrix()

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob,
                                           columncenter=columncenter)
        self.norm3 = LayerNorm(d_model=d_model, scaleonly=scaleonly)
        self.dropout3 = nn.Dropout(p=drop_prob)if dropout else nn.Identity()
        self.rescenter3 = CenteringMatrix()

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        _x = self.rescenter1(_x) if self.rescenter else nn.Identity()
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            _x = self.rescenter2(_x) if self.rescenter else nn.Identity()
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        _x = self.rescenter3(_x) if self.rescenter else nn.Identity()
        x = self.norm3(x + _x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, scaleonly=False, columncenter=False,
                 dropout=True, rescenter=False):
        super(EncoderLayer, self).__init__()
        self.rescenter = rescenter
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, columncenter=columncenter)
        self.norm1 = LayerNorm(d_model=d_model, scaleonly=scaleonly)
        self.dropout1 = nn.Dropout(p=drop_prob) if dropout else nn.Identity()
        self.rescenter1 = CenteringMatrix()

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob,
                                           columncenter=columncenter)
        self.norm2 = LayerNorm(d_model=d_model, scaleonly=scaleonly)
        self.dropout2 = nn.Dropout(p=drop_prob) if dropout else nn.Identity()
        self.rescenter2 = CenteringMatrix()

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        _x = self.rescenter1(_x) if self.rescenter else nn.Identity()
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        _x = self.rescenter2(_x) if self.rescenter else nn.Identity()
        x = self.norm2(x + _x)
        return x

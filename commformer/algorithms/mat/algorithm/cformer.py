import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
import numpy as np

from transformers.modeling_utils import Conv1D
from torch.distributions import Categorical
from commformer.algorithms.utils.util import check, init
from commformer.algorithms.utils.transformer_act import discrete_autoregreesive_act
from commformer.algorithms.utils.transformer_act import discrete_parallel_act
from commformer.algorithms.utils.transformer_act import continuous_autoregreesive_act
from commformer.algorithms.utils.transformer_act import continuous_parallel_act

class GraphTransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, n_agent, self_loop_add, dropout=0.1, weights_dropout=False, masked=False):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, n_agent, dropout, weights_dropout, masked, self_loop_add)
        self.fc1 = Conv1D(ff_embed_dim, embed_dim)
        self.fc2 = Conv1D(embed_dim, ff_embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, relation, kv=None, attn_mask=None,
                need_weights=False, dec_agent=False):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, relation=relation, attn_mask=attn_mask,
                                          need_weights=need_weights, dec_agent=dec_agent)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, relation=relation, attn_mask=attn_mask,
                                          need_weights=need_weights, dec_agent=dec_agent)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn


class RelationMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_agent, dropout=0., weights_dropout=False, masked=False, self_loop_add=True):
        super(RelationMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.masked = masked
        self.self_loop_add = self_loop_add

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = Conv1D(2 * embed_dim, embed_dim)

        self.out_proj = Conv1D(embed_dim, embed_dim)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

        self.register_buffer("mask", torch.tril(torch.ones(n_agent, n_agent)) == 0)

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, attn_mask=None, need_weights=False, dec_agent=False):
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x n_emd
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        if relation is None:
            # q *= self.scaling
            attn_weights = torch.einsum('ibn,jbn->ijb', [q, k]) * (1.0 / math.sqrt(k.size(-1)))
        else:
            ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
            ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            q = q.unsqueeze(1) + ra
            k = k.unsqueeze(0) + rb
            q *= self.scaling
            # q: tgt_len x src_len x bsz*heads x dim
            # k: tgt_len x src_len x bsz*heads x dim
            # v: src_len x bsz*heads x dim

            attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k]) * (1.0 / math.sqrt(k.size(-1)))
        
        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if self.masked:
            attn_weights.masked_fill_(
                self.mask.unsqueeze(-1),
                float('-inf')
            )

            # attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)
        
        if dec_agent is True:
            self_loop = torch.eye(tgt_len).unsqueeze(-1).long().to(device=attn_weights.device)
            if self.self_loop_add:
                attn_mask = attn_mask + self_loop
            else:
                attn_mask = attn_mask * (1 - self_loop) + self_loop
            attn_weights.masked_fill_(
                attn_mask == 0,
                float('-inf')
            )
            attn_weights = F.softmax(attn_weights, dim=1)
            attn_weights = attn_weights * attn_mask
        else:
            attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights: tgt_len x src_len x bsz*heads
        # v: src_len x bsz*heads x dim
        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.gelu(input)


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, self_loop_add):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.attn1 = RelationMultiheadAttention(n_embd, n_head, n_agent, masked=True, self_loop_add=self_loop_add)
        self.attn2 = RelationMultiheadAttention(n_embd, n_head, n_agent, masked=True, self_loop_add=self_loop_add)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc, relation_embed, attn_mask, dec_agent):
        # x : (batch, n_agent, n_embd)
        # rep_enc: (batch, n_agent, n_embd)
        # relation_embed: (batch, n_agent, n_agent, n_embed)
        bs, n_agent, n_emd = x.shape

        x_back = x.permute(1, 0, 2).contiguous() # n_agent x bs x dim
        if relation_embed is not None:
            relations_back = relation_embed.permute(1, 2, 0, 3).contiguous() # n_agent x n_agent x bs x dim
        else:
            relations_back = relation_embed
        attn_mask_back = attn_mask.permute(1, 2, 0).contiguous() # n_agent x n_agent x bs
        y, _ = self.attn1(x_back, x_back, x_back, relations_back, attn_mask=attn_mask_back, dec_agent=dec_agent)
        y = y.permute(1, 0, 2).contiguous() # bs x n_agent x dim
        
        x = self.ln1(x + y)

        rep_enc_back = rep_enc.permute(1, 0, 2).contiguous()
        x_back = x.permute(1, 0, 2).contiguous()
        y, _ = self.attn2(rep_enc_back, x_back, x_back, relations_back, attn_mask=attn_mask_back, dec_agent=dec_agent)
        y = y.permute(1, 0, 2).contiguous() # bs x 2 * n_agent x dim

        x = self.ln2(rep_enc + y)
        x = self.ln3(x + self.mlp(x))

        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, self_loop_add):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([GraphTransformerLayer(n_embd, n_embd, n_head, n_agent, self_loop_add=self_loop_add) 
                                     for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs, relation, attn_mask, dec_agent):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # relation: (bs, n_agent, n_agent, n_embed)
        # attn_mask: (bs, n_agent, n_agent)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        x = self.ln(x) # (bs, n_agent, n_emd)
        x = x.permute(1, 0, 2).contiguous() # n_agent x bs x dim
        if relation is not None:
            relation = relation.permute(1, 2, 0, 3).contiguous() # n_agent x n_agent x bs x n_embed
        attn_mask = attn_mask.permute(1, 2, 0).contiguous() # n_agent x n_agent x bs
        for idx, layer in enumerate(self.blocks):
            x, _ = layer(x, relation, attn_mask=attn_mask, dec_agent=dec_agent)
        # rep = self.blocks(self.ln(x))
        rep = x.permute(1, 0, 2).contiguous() # bs x n_agent x dim
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False, self_loop_add=True):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        
        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())
        else:
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                        init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent, self_loop_add=self_loop_add) for _ in range(n_block)])
                

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs, relation_embed, attn_mask, dec_agent):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        # relation: (batch, n_agent, n_agent)
        # fix_relation: (batch, n_agent, n_agent, n_emd)

        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep, relation_embed, attn_mask, dec_agent)
            
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(x)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](x[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            # action_embeddings = self.action_encoder(action)
            # x = self.ln(action_embeddings)
            # for block in self.blocks:
            #     x = block(x, obs_rep, relation, fix_relation)
            logit = self.head(x)

        return logit


class CommFormer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, sparsity=0.4, 
                 warmup=10, post_stable=False, post_ratio=0.5, self_loop_add=True,
                 no_relation_enhanced=False):
        super(CommFormer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tldv = dict(dtype=torch.long, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, self_loop_add)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor,
                               self_loop_add=self_loop_add)
        
        self.edges = nn.Parameter(torch.ones(n_agent, n_agent), requires_grad=True)
        self.edges_embed = nn.Embedding(2, n_embd)

        self.to(device)

        self.dec_actor = dec_actor
        self.sparsity = sparsity
        self.topk = int(max(n_agent * sparsity, 1))
        self.warmup = warmup
        self.post_stable = post_stable
        self.post_ratio = post_ratio
        self.no_relation_enhanced = no_relation_enhanced

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)
    
    def model_parameters(self):
        parameters = [p for name, p in self.named_parameters() if name != "edges" ]
        return parameters

    def edge_parameters(self):
        parameters = [p for name, p in self.named_parameters() if name == "edges"]
        return parameters

    def edge_return(self, exact=False, topk=-1):

        edges = self.edges
        if exact is False:
            relations = gumbel_softmax_topk(edges, topk=self.topk, hard=True, dim=-1)
        else:
            y_soft = edges.softmax(dim=-1)
            index = edges.topk(k=self.topk, dim=-1)[1]
            relations = torch.zeros_like(edges, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            relations = relations - y_soft.detach() + y_soft
        
        if topk != -1:
            y_soft = edges.softmax(dim=-1)
            index = edges.topk(k=topk, dim=-1)[1]
            relations = torch.zeros_like(edges, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            relations = relations - y_soft.detach() + y_soft

        return relations

    def forward(self, state, obs, action, available_actions=None, steps=0, total_step=0):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]

        if steps > self.warmup:
            # top_k
            relations = self.edge_return()
        else:
            relations = self.edges
        
        # improve the training stability
        if steps > int(self.post_ratio * total_step) and self.post_stable:
            relations = self.edge_return(exact=True)

        relations = relations.unsqueeze(0)

        relations_embed = self.edges_embed(relations.long()) 
        relations_embed = relations_embed.repeat(batch_size, 1, 1, 1) # 1 x n x n x emd

        if self.dec_actor:
            dec_agent = True
        else:
            dec_agent = False

        if self.no_relation_enhanced is True:
            relations_embed = None
            

        v_loc, obs_rep = self.encoder(state, obs, relations_embed, attn_mask=relations, dec_agent=dec_agent)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, relations_embed, relations, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions, dec_agent=dec_agent)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, relations_embed, relations, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv, dec_agent=dec_agent)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]

        relations = self.edge_return(exact=True)

        relations = relations.unsqueeze(0)
        relations_embed = self.edges_embed(relations.long()) # 1 x n x n x emd
        relations_embed = relations_embed.repeat(batch_size, 1, 1, 1)

        if self.dec_actor:
            dec_agent=True
        else:
            dec_agent=False 
        
        if self.no_relation_enhanced is True:
            relations_embed = None
        
        v_loc, obs_rep = self.encoder(state, obs, relations_embed, attn_mask=relations, dec_agent=dec_agent)

        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, relations_embed, relations, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic, dec_agent=dec_agent)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, relations_embed, relations, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic, dec_agent=dec_agent)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)

        batch_size = np.shape(obs)[0]

        relations = self.edge_return(exact=True)
        
        relations = relations.unsqueeze(0)
        relations_embed = self.edges_embed(relations.long()) # 1 x n x n x emd
        relations_embed = relations_embed.repeat(batch_size, 1, 1, 1)

        if self.dec_actor:
            dec_agent=True
        else:
            dec_agent=False 
        
        if self.no_relation_enhanced is True:
            relations_embed = None

        v_tot, obs_rep = self.encoder(state, obs, relations_embed, attn_mask=relations, dec_agent=dec_agent)
        return v_tot

def gumbel_softmax_topk(logits, topk=1, tau=1, hard=False, dim=-1):

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k=topk, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
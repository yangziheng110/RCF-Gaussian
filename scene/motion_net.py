import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class A_F_AudioNet(nn.Module):
    def __init__(self, dim_in=1024, dim_aud=64, win_size=2):  # win_size调整为2，适配固定时间维度
        super(A_F_AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 1024 x 2
            nn.Conv1d(dim_in, 512, kernel_size=1, stride=1, padding=0, bias=True),  # n x 512 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(512, 256, kernel_size=1, stride=1, padding=0, bias=True),  # n x 256 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(256, 128, kernel_size=1, stride=1, padding=0, bias=True),  # n x 128 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64 * self.win_size, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        # 直接保留整个时间维度，无需裁剪
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # 展平后输入到全连接层
        x = self.encoder_fc1(x)
        return x

class A_M_AudioNet(nn.Module):
    def __init__(self, dim_in=1024, dim_aud=64):
        super(A_M_AudioNet, self).__init__()
        self.dim_aud = dim_aud
        self.encoder_fc = nn.Sequential(  # n x 1024 x 2
            nn.Linear(dim_in * 2, 1024),  # 将时间维度合并到通道维度
            nn.LeakyReLU(0.02, True),
            nn.Linear(1024, 512),  # 将时间维度合并到通道维度
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),  # 输出维度为 dim_aud
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )

    def forward(self, x):
        # x: n x 1024 x 2
        x = x.reshape(x.size(0), -1)  # 合并时间维度，变为 n x (1024*2)
        x = self.encoder_fc(x)  # 输入到全连接网络
        return x
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadCrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim, num_heads):
        super().__init__()
        self.audio_dim = audio_dim
        self.num_heads = num_heads

        # 使用多头注意力层
        #self.multihead_attn = nn.MultiheadAttention(embed_dim=audio_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm( 32)
        
        self.ffn = PositionwiseFeedForward(d_model= 32, hidden= 64)
        self.norm2 = nn.LayerNorm( 32)
        

       

    def forward(self, audio_feat, spatial_att):
        """
        Args:
            audio_feat: [1, audio_dim] (音频特征，1 表示批量大小)
            spatial_att: [N, audio_dim] (空间特征，N 是点的数量)
        Returns:
            out: [N, audio_dim]
        """
        # 将音频特征扩展到与空间特征匹配的长度
        #audio_feat_expanded = audio_feat.expand(spatial_att.size(0), -1)  # [N, audio_dim]
        x = audio_feat.repeat(spatial_att.shape[0], 1)
        # 输入 MultiheadAttention 的格式需要 (Batch_Size, Seq_Len, Dim)，因此调整维度
        #query = spatial_att.unsqueeze(0)  # [1, N, audio_dim]
        #key_value = audio_feat.unsqueeze(0)  # [1, N, audio_dim]

        # 计算多头注意力
        #attn_output, attn_weights = self.multihead_attn(query, key_value, key_value)  # attn_output: [1, N, audio_dim]
        attn_output= x*spatial_att
        # 去掉 batch 维度
        #out = attn_output.squeeze(0)  # [N, audio_dim]
        out = attn_output
        out = self.norm1(x + out)
        _out = out
        out = self.ffn( out )
        #out = self.dropout2(out)
        out = self.norm2(out + _out)
       

        return out



class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x

class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = A_F_AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 4     
        self.hidden_dim = 80

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
        self.MultiHeadCrossAttentionFusion = MultiHeadCrossAttentionFusion(32,4)
        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None
        enc_a = self.audio_net(a) # [1/8, 64]
        #print("1",enc_a.shape)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
        #print("2",enc_a.shape) 
        return enc_a


    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        #print(enc_a.shape)
        
       
        aud_ch_att = self.aud_ch_att_net(enc_x)

        h_x_a = self.MultiHeadCrossAttentionFusion(enc_a ,aud_ch_att)
        #print(aud_ch_att.shape)
        #scores_x_a = torch.matmul(aud_ch_att, enc_a.transpose(-2, -1)) / (aud_ch_att.size(-1) ** 0.5)
        
        #attention_weights_x_a = F.softmax(scores_x_a, dim=-1)
        
        #enc_a_repeat = enc_a.repeat(enc_x.shape[0], 1)
       
    # 计算加权和
        #h_x_a = torch.matmul(attention_weights_x_a, enc_a)
        #h_x_a = h_x_a*0.5 + enc_a*0.5
        #enc_a = enc_a.repeat(enc_x.shape[0], 1)
        #enc_w = enc_a_repeat * aud_ch_att
        eye_att = torch.relu(self.eye_att_net(enc_x))
        #print(eye_att.shape)
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e =enc_e.unsqueeze(0)
        #enc_e = enc_e * eye_att
        #print(enc_e.shape)
        scores_x_e = torch.matmul(eye_att, enc_e.transpose(-2, -1)) / (eye_att.size(-1) ** 0.5)
        attention_weights_x_e = F.softmax(scores_x_e, dim=-1)
        h_x_e = torch.matmul(attention_weights_x_e, enc_e)
        #h_x_e = h_x_e*0.5 + enc_e*0.5
        #if c is not None:
            #c = c.repeat(enc_x.shape[0], 1)
            #h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        #else:
        h = torch.cat([enc_x,h_x_a, h_x_e], dim=-1)
        

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.MultiHeadCrossAttentionFusion.parameters(), 'lr': lr_net*2  , 'weight_decay': 0.0001})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params




class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = A_M_AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)
        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        #aud_ch_att = self.aud_ch_att_net(enc_x)
        #scores_x_a = torch.matmul(enc_a, aud_ch_att.transpose(-2, -1)) / (aud_ch_att.size(-1) ** 0.5)
        #attention_weights_x_a = F.softmax(scores_x_a, dim=-1)
        
        #attention_weights_x_a = attention_weights_x_a.squeeze(0).unsqueeze(-1)
        #h_x_a = aud_ch_att * attention_weights_x_a
        #enc_w = enc_a * aud_ch_att
        #scores_x_a = torch.matmul(aud_ch_att, enc_a.transpose(-2, -1)) / (aud_ch_att.size(-1) ** 0.5)
        
        #attention_weights_x_a = F.softmax(scores_x_a, dim=-1)
        
       
       
    # 计算加权和
        #h_x_a = torch.matmul(attention_weights_x_a, enc_a)
        h = torch.cat([enc_x,enc_w], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params

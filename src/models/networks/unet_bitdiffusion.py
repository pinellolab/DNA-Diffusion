# each layer has a time embedding AND class conditioned embedding

class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        bits = BITS,
        resnet_block_groups = 8,
        learned_sinusoidal_dim = 16,
        num_classes=10,
        class_embed_dim=3,
    ):
        super().__init__()

        # determine dimensions

        channels *= bits #lucas
        self.channels = channels *2

        input_channels = channels * 2
        #input_channels =16

        
        init_dim = default(init_dim, dim)
        #self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3) # original TODO for zach: is there a difference?
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7,7), padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]


        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlockClassConditioned, groups=resnet_block_groups,
                              num_classes=num_classes, class_embed_dim=class_embed_dim)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)

        #self.final_conv = nn.Conv2d(dim, 1, 1) #lucas
        self.final_conv = nn.Conv2d(dim,8, 1)


    def forward(self, x, time, c, x_self_cond = None):
        #print(x.shape)
        #c = torch.zeros_like(c) # removing the conditioning LUCAS

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        
        x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        
        # todo class mask

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)

            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        
        x = self.final_conv(x)
        #print(x.shape, 'final')
        return x
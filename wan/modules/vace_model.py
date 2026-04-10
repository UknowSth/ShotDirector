# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanModel(WanModel):

    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                start_masked_layer=18,
                 end_masked_layer=26,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)
                           ] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {
            i: n for n, i in enumerate(self.vace_layers)
        }

        self.start_masked_layer = start_masked_layer
        self.end_masked_layer = end_masked_layer

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=self.vace_layers_mapping[i]
                if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=i) for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block in self.vace_blocks:
            c, c_skip = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        premask=None,
        premask_text=None,
        camera_info=None,
        camera_info_recam=None,
        use_camera=None,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        _, h, w = x[0].shape[-3], x[0].shape[-2], x[0].shape[-1] # wxx
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if camera_info is not None:
            if camera_info.ndim == 3: # param
                camera_info = camera_info.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
                camera_info = rearrange(camera_info, 'b f h w d -> b (f h w) d')
                # print(f'cam_emb:{camera_info.shape}',flush=True) 
            if camera_info.ndim == 5: # plucker
                camera_info = rearrange(camera_info, 'b f h w d -> b d f h w')
                # print(f'camera_info:{camera_info.shape}',flush=True)
                camera_info = self.patch_embedding_cam(camera_info)
                camera_info = rearrange(camera_info, 'b d f h w -> b (f h w) d')
        if camera_info_recam is not None:
            if camera_info_recam.ndim == 3: # param
                camera_info_recam = camera_info_recam.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
                camera_info_recam = rearrange(camera_info_recam, 'b f h w d -> b (f h w) d')
                # print(f'cam_emb:{camera_info.shape}',flush=True) 

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # context = self.text_embedding(
        #     torch.stack([
        #         torch.cat(
        #             [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        #         for u in context
        #     ]))
        context = [self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in v
                ])
            ) for v in context] # 没法凑长度一样
        # 接下来解决每个视频的shot数不一样的问题 → 直接在构建batch的时候增加"",并且在mask中提前规划好,一定可以确保shot数都相同
        context = torch.concat([u.reshape(1, -1, self.dim) for u in context], dim=0)

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            attn_mask=premask,
            attn_mask_text=premask_text,
            camera_info=camera_info,
            camera_info_recam=camera_info_recam,
            use_camera=use_camera)
        kwargs1 = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            attn_mask=None,
            attn_mask_text=None,
            camera_info=camera_info,
            camera_info_recam=camera_info_recam,
            use_camera=use_camera)

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale
        kwargs1['hints'] = hints
        kwargs1['context_scale'] = vace_context_scale

        for ii, block in enumerate(self.blocks):
            if ii >= self.start_masked_layer and ii <= self.end_masked_layer:
                x = block(x, **kwargs)
            else:
                x = block(x, **kwargs1)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
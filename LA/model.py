"""
Author: Haoran Chen
Date: 2022.08.15
"""

import torch
from clip_custom import clip

from torch import nn
from einops import rearrange


class PromptGenerator(nn.Module):
    def __init__(self, classnames, clip_model, source_names, target_name, args):
        super().__init__()
        n_cls = len(classnames)
        dtype = torch.float32
        embedding_dim = clip_model.ln_final.weight.shape[0]
        ctx_cls_vector = torch.empty(
            n_cls,
            args.M1,
            embedding_dim,
            requires_grad=True,
            dtype=dtype,
            device=args.device,
        )
        ctx_source_vector = torch.empty(
            1,
            args.M2,
            embedding_dim,
            requires_grad=True,
            dtype=dtype,
            device=args.device,
        )
        ctx_target_vector = torch.empty(
            1,
            args.M2,
            embedding_dim,
            requires_grad=True,
            dtype=dtype,
            device=args.device,
        )

        nn.init.normal_(ctx_cls_vector, std=0.02)
        nn.init.normal_(ctx_source_vector, std=0.02)
        nn.init.normal_(ctx_target_vector, std=0.02)

        prompt_prefix = " ".join(["X"] * (args.M1 + args.M2))

        self.ctx_cls = nn.Parameter(ctx_cls_vector)  # to be optimized
        self.ctx_source = nn.Parameter(ctx_source_vector)  # to be optimized
        self.ctx_target = nn.Parameter(ctx_target_vector)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            args.device
        )

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOT
        self.register_buffer(
            "token_suffix", embedding[:, 1 + args.M1 + args.M2 :, :]
        )  # CLS, EOT

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts

    def forward(self):

        ctx_cls = self.ctx_cls
        ctx_target = self.ctx_target

        prefix = self.token_prefix
        suffix = self.token_suffix

    def forward(self):
        ctx_cls = self.ctx_cls
        ctx_source = self.ctx_source
        ctx_target = self.ctx_target

        prefix = self.token_prefix
        suffix = self.token_suffix

        source_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_cls,  # (n_cls, M1, dim)
                ctx_source.repeat(self.n_cls, 1, 1),  # (n_cls, 1, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        target_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_cls,
                ctx_target.repeat(self.n_cls, 1, 1),  # (n_cls, 1, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return source_prompts, target_prompts

    def get_source_grad(self, zero_grad=True):
        grad = self.ctx_source.grad.data.clone().flatten()
        if zero_grad:
            self.ctx_source.grad.zero_()
        return grad

    def set_source_grad(self, grad):
        self.ctx_source.grad = grad.data.clone().reshape(self.ctx_source.shape)

    def get_target_grad(self, zero_grad=True):
        grad = self.ctx_target.grad.data.clone().flatten()
        if zero_grad:
            self.ctx_target.grad.zero_()
        return grad

    def set_target_grad(self, grad):
        self.ctx_target.grad = grad.data.clone().reshape(self.ctx_target.shape)

    def get_shared_grad(self, zero_grad=True):
        grad = self.ctx_cls.grad.data.clone().flatten()
        if zero_grad:
            self.ctx_cls.grad.zero_()
        return grad

    def set_shared_grad(self, grad):
        self.ctx_cls.grad = grad.data.clone().reshape(self.ctx_cls.shape)

    def get_source_param(self):
        return self.ctx_source.data.clone().flatten()

    def set_source_param(self, params):
        self.ctx_source.data = params.data.clone().reshape(self.ctx_source.shape)

    def get_target_param(self):
        return self.ctx_target.data.clone().flatten()

    def set_target_param(self, params):
        self.ctx_target.data = params.data.clone().reshape(self.ctx_target.shape)

    def get_shared_param(self):
        return self.ctx_cls.data.clone().flatten()

    def set_shared_param(self, params):
        self.ctx_cls.data = params.data.clone().reshape(self.ctx_cls.shape)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype

    def forward(self, prompts, tokenized_prompts):
        prompts = prompts.type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class Custom_Clip(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, prompt, tokenized_prompts):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(prompt, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


class AutoEncoder(nn.Module):
    def __init__(self, dim, decoder_dim, inner_dim):
        super().__init__()
        self.prompt_w1 = nn.Linear(dim, inner_dim)
        self.prompt_w2 = nn.Linear(inner_dim, decoder_dim)
        self.prompt_w3 = nn.Linear(decoder_dim, dim)

    def forward(self, x):
        batch = x.size()[0]
        x = rearrange(x, "b t e -> (b t) e")
        x = x.to(self.prompt_w1.weight.device)
        x = self.prompt_w1(x)
        x = torch.tanh(self.prompt_w2(x))
        x = self.prompt_w3(x)
        x = rearrange(x, "(b t) e -> b t e", b=batch)
        return x

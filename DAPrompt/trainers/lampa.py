import os.path as osp
import os
import datetime
import time

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchinfo import summary

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.LAMPA.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert len(cfg.DATASET.TARGET_DOMAINS) == 1, "Only one target domain is supported"
        
        src_domains = cfg.DATASET.SOURCE_DOMAINS
        tgt_domain = cfg.DATASET.TARGET_DOMAINS[0]
        if tgt_domain in src_domains :
            raise ValueError("Target domain should not be in source domains")
        self.n_srcs = len(src_domains)
        n_dm = 1  # number of domains
        n_dmx = cfg.TRAINER.LAMPA.N_DMX  # number of domain context
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = "a photo of a".replace("_", " ")

        if cfg.TRAINER.LAMPA.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n)
        classnames = [name.replace("_", " ") for name in classnames]
        self.srcs = []
        self.src_tokenized_prompts = []
        
        for src_domain in src_domains:
            source_vector = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
            nn.init.normal_(source_vector, std=0.02)
            self.srcs.append(nn.Parameter(source_vector))

            src_prompt = [
                prompt_prefix + " " + name + ", a " + src_domain.replace("_", " ") +"image."
                for name in classnames
            ]

            src_tokenized_prompt = torch.cat([clip.tokenize(p) for p in src_prompt])
            self.src_tokenized_prompts.append(src_tokenized_prompt)
        self.srcs = nn.ParameterList(self.srcs)   
        target_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(target_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.tgt = nn.Parameter(target_vectors)  # to be optimized
        
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]
        
        tgt_prompts = [
            prompt_prefix + " " + name + ", a " + tgt_domain.replace("_", " ") +"image."
            for name in classnames
        ]
        tgt_tokenized_prompts = torch.cat([clip.tokenize(p) for p in tgt_prompts])
        naive_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            src_embeddings = []
            for src_id, src_domain in enumerate(src_domains):
                
                src_embedding = clip_model.token_embedding(self.src_tokenized_prompts[src_id]).type(
                    dtype)
                src_embeddings.append(src_embedding)
                
            tgt_embedding = clip_model.token_embedding(tgt_tokenized_prompts).type(
                dtype)
            naive_embedding = clip_model.token_embedding(
                naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        for src_id, src_domain in enumerate(src_domains):
            self.src_tokenized_prompts[src_id] = torch.cat([self.src_tokenized_prompts[src_id], naive_tokenized_prompts])
        self.register_buffer("src_token_prefix", torch.stack([src_embedding[:, :1, :] for src_embedding in src_embeddings]))  # SOS
        self.register_buffer("src_token_suffix", torch.stack([src_embedding[:, 1 + n:, :] for src_embedding in src_embeddings]))  # CLS, EOS
        
        tgt_tokenized_prompts = torch.cat([tgt_tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("tgt_token_prefix", tgt_embedding[:, :1, :])  # SOS
        self.register_buffer("tgt_token_suffix", tgt_embedding[:, 1 + n:, :])
        self.tgt_tokenized_prompts = tgt_tokenized_prompts

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.LAMPA.CSC
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.to(torch.device("cuda"))

    @autocast()
    def forward(self, src_id=None):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        if  src_id is not None:
            dmx = self.srcs[src_id]
            prefix = self.src_token_prefix[src_id]
            suffix = self.src_token_suffix[src_id]    
        else:
            dmx = self.tgt
            prefix = self.tgt_token_prefix
            suffix = self.tgt_token_suffix
            
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1,
                                              -1)  # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1,
                                          -1)  # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx], dim=2).reshape(self.n_cls * self.n_dm,
                                          self.n_ctx + self.n_dmx, ctx_dim)

        # naive
        neb = self.naive_embedding
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctxdmx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat([prompts, neb], dim=0)
        return prompts
    def get_source_grad(self, src_id, zero_grad=True):
        grad = self.srcs[src_id].grad.data.clone().flatten()
        if zero_grad:
            self.srcs[src_id].grad.zero_()
        return grad

    def set_source_grad(self, grad, src_id):
        self.srcs[src_id].grad = grad.data.clone().reshape(self.srcs[src_id].shape)

    def get_target_grad(self, zero_grad=True):
        grad = self.tgt.grad.data.clone().flatten()
        if zero_grad:
            self.tgt.grad.zero_()
        return grad

    def set_target_grad(self, grad):
        self.tgt.grad = grad.data.clone().reshape(self.tgt.shape)

    def get_shared_grad(self, zero_grad=True):
        grad = self.ctx.grad.data.clone().flatten()
        if zero_grad:
            self.ctx.grad.zero_()
        return grad

    def set_shared_grad(self, grad):
        self.ctx.grad = grad.data.clone().reshape(self.ctx.shape)

    def get_source_param(self, src_id):
        return self.srcs[src_id].data.clone().flatten()

    def set_source_param(self, params, src_id):
        self.srcs[src_id].data = params.data.clone().reshape(self.srcs[src_id].shape)

    def get_target_param(self):
        return self.tgt.data.clone().flatten()

    def set_target_param(self, params):
        self.tgt.data = params.data.clone().reshape(self.tgt.shape)

    def get_shared_param(self):
        return self.ctx.data.clone().flatten()

    def set_shared_param(self, params):
        self.ctx.data = params.data.clone().reshape(self.ctx.shape)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.src_tokenized_prompts = self.prompt_learner.src_tokenized_prompts
        self.tgt_tokenized_prompts = self.prompt_learner.tgt_tokenized_prompts
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, image, src_id=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner(src_id=src_id)
        if src_id is None:
            tokenized_prompts = self.tgt_tokenized_prompts
        else:
            tokenized_prompts = self.src_tokenized_prompts[src_id]
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    


@TRAINER_REGISTRY.register()
class LAMPA(TrainerXU):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LAMPA.PREC in ["fp16", "fp32", "amp"]
    
    def model_inference(self, input, ensemble=False):
        
        output_tgt = self.model(input, src_id=None)
        if not ensemble:
            return output_tgt
        for src_id in range(self.n_srcs):
            output_src = self.model(input, src_id=src_id)
            output_tgt += output_src
        output = (output_src + output_tgt) / (self.n_srcs + 1)
        return output
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_srcs = len(cfg.DATASET.SOURCE_DOMAINS)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LAMPA.PREC == "fp32" or cfg.TRAINER.LAMPA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # plus one for pseudo label
        self.n_dm = self.model.prompt_learner.n_dm + 1
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            print(name, param.shape, param.requires_grad)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LAMPA.PREC == "amp" else None

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""

        self.before_train()
        # summary(self.model, input_size=(1, 3, 224, 224))
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.epoch>50:
                break
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        # self.test_batches = [int(self.num_batches * 0.33), int(self.num_batches * 0.66)]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        # label_u only used for matric
        image_x, label, image_u, domain = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.LAMPA.PREC
        if prec == "amp":
            with autocast():

                shared_param = self.model.prompt_learner.get_shared_param()
                source_params = []
                for src_id in range(self.n_srcs):
                    source_param = self.model.prompt_learner.get_source_param(src_id)
                    source_params.append(source_param)
                target_param = self.model.prompt_learner.get_target_param()

                # loss 1, stage 1 #
                enable_running_stats(self.model)
                self.optim.zero_grad()
                output_u = self.model(image_u, src_id=None)
                
                # only clip annotation
                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1, self.n_cls) /
                    self.cfg.TRAINER.LAMPA.T,
                    dim=-1)

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.cfg.TRAINER.LAMPA.TAU).float()

                loss_u = (F.cross_entropy(
                    output_u[:, :self.n_cls],
                    label_p,
                    reduction="none") * mask).sum() / mask.sum()
                
                self.scaler.scale(loss_u).backward()                
                target_grad = self.model.prompt_learner.get_target_grad()
                shared_grad_tgt = self.model.prompt_learner.get_shared_grad()
                
                
                
                # loss 2, stage 1 #
                source_grads = []
                shared_grads_srcs = []
                for src_id in range(self.n_srcs):
                    enable_running_stats(self.model)
                    self.optim.zero_grad()
                    output_x = self.model(image_x[domain==src_id], src_id=src_id)
                    loss_x = F.cross_entropy(output_x[:, :self.n_cls], label[domain==src_id])
                    self.scaler.scale(loss_x).backward()                   
                    source_grad = self.model.prompt_learner.get_source_grad(src_id)
                    shared_grad_src = self.model.prompt_learner.get_shared_grad()   
                    source_grads.append(source_grad)
                    shared_grads_srcs.append(shared_grad_src) 
                
                source_grads = torch.stack(source_grads)
                shared_grads_srcs = torch.stack(shared_grads_srcs)
                
                loss = loss_x + loss_u
                
                # loss 1, stage 2 #
                perturbed_target_param = target_param + self.cfg.TRAINER.LAMPA.radius * target_grad / (
                    torch.norm(target_grad) + 1e-12
                )
                self.model.prompt_learner.set_target_param(perturbed_target_param)

                perturbed_shared_param = shared_param + self.cfg.TRAINER.LAMPA.radius * shared_grad_tgt / (
                    torch.norm(shared_grad_tgt) + 1e-12
                ) - self.cfg.TRAINER.LAMPA.align * sum([shared_grad_src/(torch.norm(shared_grad_src)*torch.norm(shared_grad_tgt) + 1e-12) for shared_grad_src in shared_grads_srcs])
                
                self.model.prompt_learner.set_shared_param(perturbed_shared_param)   
                disable_running_stats(self.model)
                output_u = self.model(image_u, src_id=None)
                
                # only clip annotation
                pseudo_label = torch.softmax(
                    output_u[:, -self.n_cls:].reshape(-1, self.n_cls) /
                    self.cfg.TRAINER.LAMPA.T,
                    dim=-1)

                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.cfg.TRAINER.LAMPA.TAU).float()

                loss_u = (F.cross_entropy(
                    output_u[:, :self.n_cls],
                    label_p,
                    reduction="none") * mask).sum() / mask.sum()
                
                self.scaler.scale(loss_u).backward()  
                target_grad = self.model.prompt_learner.get_target_grad()
                shared_grad_tgt_new = self.model.prompt_learner.get_shared_grad()

                # loss 2, stage 2 #
                source_grads_new = []
                shared_grad_srcs_new = []
                for src_id in range(self.n_srcs):
                    source_param = source_params[src_id]
                    source_grad = source_grads[src_id]
                    shared_grad_src = shared_grads_srcs[src_id]
                    
                    perturbed_source_param = source_param + self.cfg.TRAINER.LAMPA.radius * source_grad / (
                        torch.norm(source_grad) + 1e-12
                    )
                    self.model.prompt_learner.set_source_param(perturbed_source_param, src_id)
                    
                    perturbed_shared_param = shared_param + self.cfg.TRAINER.LAMPA.radius * shared_grad_src / (
                        torch.norm(shared_grad_src) + 1e-12 
                    ) - self.cfg.TRAINER.LAMPA.align * shared_grad_tgt/(torch.norm(shared_grad_tgt)*torch.norm(shared_grad_src) + 1e-12)
                    self.model.prompt_learner.set_shared_param(perturbed_shared_param)
                    
                    
                    disable_running_stats(self.model)
                    self.optim.zero_grad()
                    output_x = self.model(image_x[domain==src_id], src_id=src_id)
                    loss_x = F.cross_entropy(output_x[:, :self.n_cls], label[domain==src_id])
                    self.scaler.scale(loss_x).backward()     
                    
                    source_grad = self.model.prompt_learner.get_source_grad(src_id)
                    shared_grad_src_new = self.model.prompt_learner.get_shared_grad()
                    source_grads_new.append(source_grad)
                    shared_grad_srcs_new.append(shared_grad_src_new)

                # loss update #
                # torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), 1)
                self.model.prompt_learner.set_target_param(target_param)
                for src_id in range(self.n_srcs):
                    self.model.prompt_learner.set_source_param(source_params[src_id], src_id)
                    self.model.prompt_learner.set_source_grad(source_grads_new[src_id], src_id)
                    
                self.model.prompt_learner.set_shared_param(shared_param)            

                shared_grad = shared_grad_tgt_new + sum([shared_grad_src_new for shared_grad_src_new in shared_grad_srcs_new]) * self.cfg.TRAINER.LAMPA.tradeoff
                self.model.prompt_learner.set_shared_grad(shared_grad)
                self.model.prompt_learner.set_target_grad(target_grad)
                

            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()

        loss_summary = {
            "loss":
            loss.item(),
            "loss_x":
            loss_x.item(),
            "loss_u":
            loss_u.item(),
        }

        self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            # print("Testing the test prompt")
            # curr_result = self.test(ensemble=False)
            # is_best = curr_result > self.best_result
            # if is_best:
            #     self.best_result = curr_result
            #     self.save_model(self.epoch,
            #                     self.output_dir,
            #                     model_name="model-best.pth.tar")
            print("Testing the ensemble prompt")
            curr_result = self.test(ensemble=True)
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")
            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        domain = batch_x["domain"]
        
        input_u = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        domain = domain.to(self.device)
        return input, label, input_u, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None, ensemble=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, ensemble=ensemble).reshape(
                -1, self.n_dm, self.n_cls)
            # the last second slice is the logits for target domain
            output = output[:, 0, :]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
import os
import argparse
import numpy as np
import tqdm
import sys
import matplotlib.pyplot as plt

from torchinfo import summary
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from dataloader import load_pseudo_label_data, load_data
from clip_custom import clip
from model import Custom_Clip, PromptGenerator
from utils import disable_running_stats, enable_running_stats
from dataset import SingleSourceDataset
from samplers import RandomDomainSampler


def arg_parse():
    parser = argparse.ArgumentParser("Training and Evaluation Script", add_help=False)

    # for config
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"/vast/data/office-31/",
        help="data file path",
    )
    parser.add_argument("--backbone", type=str, default="RN101", help="")
    parser.add_argument("--dataset", type=str, default="ImageCLEF", help="")
    parser.add_argument("--seed", type=int, default=1, help="")

    # for dataloader
    parser.add_argument("--batch_size", type=int, default=30, help="")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--pin_memory", type=bool, default=True, help="")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="threshold tau for generating pseudo labels",
    )

    # for prompt settings
    parser.add_argument(
        "--M1", type=int, default=16, help="number of classification tokens"
    )
    parser.add_argument("--M2", type=int, default=16, help="number of domain tokens")

    # for training settings
    parser.add_argument("--prompt_iteration", type=int, default=5000, help="")
    parser.add_argument("--prompt_learning_rate", type=float, default=0.003, help="")
    parser.add_argument("--radius", type=float, default=0.1, help="")
    parser.add_argument("--align", type=float, default=0.005, help="")
    parser.add_argument("--tradeoff", type=float, default=0.5, help="")
    parser.add_argument("--entropy_tradeoff", type=float, default=0.5, help="")

    return parser


def entropy_loss(logits):
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(p * log_p, dim=-1)
    return loss.mean()

def args_update(args):
    if args.dataset == "ImageCLEF":
        args.backbone = "RN50"
        args.prompt_iteration = 400

    if args.dataset == "Office31":
        args.backbone = "RN50"
        args.prompt_iteration = 600

    if args.dataset == "DomainNet":
        args.backbone = "RN101"
        args.prompt_iteration = 4000

    if args.dataset == "OfficeHome":
        args.backbone = "RN50"
        args.prompt_iteration = 1000

    if args.dataset == "PACS":
        args.backbone = "RN18"
        args.prompt_iteration = 800

def test(target_test_loader, custom_clip_model, prompt_list, tokenized_prompts, args):
    scale = custom_clip_model.logit_scale.exp()

    correct = 0
    tot = 0
    with torch.no_grad():
        for data, label in target_test_loader:
            tot += args.batch_size
            data = data.to(args.device)
            label = label.to(args.device)

            tot_logits = 0

            # TODO: test on multiple prompts
            
            for prompt in prompt_list:
                img_feature, txt_feature = custom_clip_model(
                    data, prompt, tokenized_prompts
                )
                logits = scale * img_feature @ txt_feature.t()
                logits = logits.softmax(dim=-1)
                tot_logits += logits

            tot_logits /= len(prompt_list)
            output = torch.argmax(tot_logits, dim=1)

            correct += (output == label).sum().item()

        # print("accuracy is: {} with a total of {} data".format(correct / tot, tot))

    return correct / tot


def train(domain_list, classnames, clip_model, preprocess, args):
    custom_clip_model = Custom_Clip(clip_model)
    custom_clip_model = nn.DataParallel(custom_clip_model)
    custom_clip_model = custom_clip_model.module
    
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    for name, param in custom_clip_model.named_parameters():
        param.requires_grad_(False)
    print("Custom_Clip", summary(custom_clip_model))
    best_accs = []
    for target_name in domain_list:
        print("*" * 50)
        print("Start training on {}".format(target_name))
        if args.dataset == "DomainNet":
            if target_name not in ['quickdraw']:
                continue
        if target_name in ["b", "i", "p"]:
            continue
        tgt_save_path = os.path.join(args.output_dir, target_name)
        os.makedirs(tgt_save_path, exist_ok=True)
        result_path = os.path.join(tgt_save_path, "best_accuracy.txt")
        if os.path.exists(result_path):
            continue
        orig_stdout = sys.stdout
        f = open(tgt_save_path+ "/train.log", "w+")
        sys.stdout = f

        source_name_list = domain_list.copy()
        source_name_list.remove(target_name)
        
        # target_path = os.path.join(args.data_root, args.dataset, target_name)
        target_path = os.path.join(args.data_root, target_name)

        target_train_loader = load_pseudo_label_data(
            target_name, target_path, preprocess, clip_model, args
        )
        target_test_loader = load_data(target_path, preprocess, args)

        source_train_dataset = SingleSourceDataset(
            args.data_root, source_name_list, preprocess
        )
        sampler = RandomDomainSampler(
            source_train_dataset.data, args.batch_size, len(source_name_list)
        )
        source_train_loader = torch.utils.data.DataLoader(
            source_train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        scale = custom_clip_model.logit_scale.exp()
        prompt_learner = PromptGenerator(
            classnames, clip_model, source_name_list, target_name, args
        )
        print("PromptGenerator", summary(prompt_learner))
        tokenized_prompts = prompt_learner.tokenized_prompts

        optimizer = torch.optim.AdamW(
            list(prompt_learner.parameters()), lr=args.prompt_learning_rate
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.prompt_iteration)

        for name, param in prompt_learner.named_parameters():
            print(
                f"name: {name}, shape {param.shape}, require grad: {param.requires_grad}"
            )
        best_acc = 0
        n_conflict = 0
        grad_cosine = []
        running_avg_cosine = 0
        pbar = tqdm.tqdm(range(1, args.prompt_iteration + 1))
        for step in pbar:
            source_prompts, target_prompts = prompt_learner()

            try:
                target_data, target_label = next(target_iter)
            except Exception as err:
                target_iter = iter(target_train_loader)
                target_data, target_label = next(target_iter)

            try:
                source_data, source_label = next(source_iter)
            except Exception as err:
                source_iter = iter(source_train_loader)
                source_data, source_label = next(source_iter)

            target_data = target_data.to(args.device)
            target_label = target_label.to(args.device)
            source_data = source_data.to(args.device)
            source_label = source_label.to(args.device)

            # print("target_data", target_data.shape)
            # print("source_data", source_data.shape)
            # print("target_label", target_label.shape)
            # print("source_label", source_label.shape)

            shared_param = prompt_learner.get_shared_param()
            source_param = prompt_learner.get_source_param()
            target_param = prompt_learner.get_target_param()

            # loss 1, stage 1 #
            optimizer.zero_grad()
            enable_running_stats(custom_clip_model)
            target_img_features, target_txt_features = custom_clip_model(
                target_data, target_prompts, tokenized_prompts
            )
            target_logits = scale * target_img_features @ target_txt_features.t()
            # cross entropy loss for those that have non -1 labels
            target_cls_loss = F.cross_entropy(
                target_logits[target_label != -1], target_label[target_label != -1]
            )
            target_entropy_loss = entropy_loss(target_logits[target_label == -1])
            target_loss = target_cls_loss + args.entropy_tradeoff * target_entropy_loss
            target_loss.backward()

            target_grad = prompt_learner.get_target_grad()
            shared_grad_tgt = prompt_learner.get_shared_grad()

            # loss 2, stage 1 #
            enable_running_stats(custom_clip_model)
            optimizer.zero_grad()
            source_img_features, source_txt_features = custom_clip_model(
                source_data, source_prompts, tokenized_prompts
            )
            source_logits = scale * source_img_features @ source_txt_features.t()
            source_loss = F.cross_entropy(source_logits, source_label)
            source_loss.backward()

            source_grad = prompt_learner.get_source_grad()
            shared_grad_src = prompt_learner.get_shared_grad()
            
            similarity = cos(shared_grad_src, shared_grad_tgt).item()
            running_avg_cosine = step/(step+1) * running_avg_cosine + 1/(step+1) * similarity
            
            grad_cosine.append(running_avg_cosine)
            n_conflict += similarity < 0      
                  
            # loss 1, stage 2 #
            perturbed_target_param = target_param + args.radius * target_grad / (
                torch.norm(target_grad) + 1e-12
            )
            prompt_learner.set_target_param(perturbed_target_param)

            perturbed_shared_param = shared_param + args.radius * shared_grad_tgt / (
                torch.norm(shared_grad_tgt) + 1e-12
            ) - args.align * shared_grad_src/(torch.norm(shared_grad_src)*torch.norm(shared_grad_tgt) + 1e-12)
            
            prompt_learner.set_shared_param(perturbed_shared_param)
            
            source_prompts, target_prompts = prompt_learner()
            disable_running_stats(custom_clip_model)
            target_img_features, target_txt_features = custom_clip_model(
                target_data, target_prompts, tokenized_prompts
            )
            target_logits = scale * target_img_features @ target_txt_features.t()
            target_cls_loss = F.cross_entropy(
                target_logits[target_label != -1], target_label[target_label != -1]
            )
            target_entropy_loss = entropy_loss(target_logits[target_label == -1])
            target_loss = target_cls_loss + args.entropy_tradeoff * target_entropy_loss
            target_loss.backward()
            
            target_grad = prompt_learner.get_target_grad()
            shared_grad_tgt_new = prompt_learner.get_shared_grad()
            
            # loss 2, stage 2 #
            perturbed_source_param = source_param + args.radius * source_grad / (
                torch.norm(source_grad) + 1e-12
            )
            prompt_learner.set_source_param(perturbed_source_param)
            
            perturbed_shared_param = shared_param + args.radius * shared_grad_src / (
                torch.norm(shared_grad_src) + 1e-12 
            ) - args.align * shared_grad_tgt/(torch.norm(shared_grad_tgt)*torch.norm(shared_grad_src) + 1e-12)
            prompt_learner.set_shared_param(perturbed_shared_param)
    
            source_prompts, target_prompts = prompt_learner()
            disable_running_stats(custom_clip_model)
            optimizer.zero_grad()
            source_img_features, source_txt_features = custom_clip_model(
                source_data, source_prompts, tokenized_prompts
            )
            source_logits = scale * source_img_features @ source_txt_features.t()
            source_loss = F.cross_entropy(source_logits, source_label)
            source_loss.backward()

            source_grad = prompt_learner.get_source_grad()
            shared_grad_src_new = prompt_learner.get_shared_grad()
            

            # loss update #
            # torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), 1)
            prompt_learner.set_target_param(target_param)
            prompt_learner.set_source_param(source_param)
            prompt_learner.set_shared_param(shared_param)            

            shared_grad = shared_grad_tgt_new + shared_grad_src_new * args.tradeoff
            prompt_learner.set_shared_grad(shared_grad)
            prompt_learner.set_target_grad(target_grad)
            prompt_learner.set_source_grad(source_grad)
            optimizer.step()

            if step % (args.prompt_iteration / 20) == 0:
                scheduler.step()

            # prompt_list = [target_prompts]
            # acc = test(
            #     target_test_loader,
            #     custom_clip_model,
            #     prompt_list,
            #     tokenized_prompts,
            #     args,
            # )
            # pbar.set_description(
            #     f"step: {step}, accuracy: {acc}, target total loss: {target_loss.item()}, classification: {target_cls_loss.item()}, entropy: {target_entropy_loss.item()}"
            # )
            # if acc > best_acc:
            #     best_acc = acc
            # print(f"Best accuracy so far: {best_acc}, step {step}, accuracy {acc}")
            if args.dataset == "DomainNet":
                if step < 3000:
                    if step%500:
                        continue
                else:
                    if step%10:
                        continue
            prompt_list = [source_prompts, target_prompts]
            acc = test(
                target_test_loader,
                custom_clip_model,
                prompt_list,
                tokenized_prompts,
                args,
            )
            pbar.set_description(
                f"step: {step}, accuracy: {acc}, target total loss: {target_loss.item()}, classification: {target_cls_loss.item()}, entropy: {target_entropy_loss.item()}"
            )
            if acc > best_acc:
                best_acc = acc
            print(f"Best accuracy so far: {best_acc}, step {step}, accuracy {acc}")
        best_accs.append(best_acc)
        print("Best accuracy for each domain:", best_accs, "Average:", np.mean(best_accs))
        print("Number of conflicts:", n_conflict, "Total steps:", args.prompt_iteration, "Conflict rate:", n_conflict/args.prompt_iteration)
        print("Average cosine similarity between gradients:", np.mean(grad_cosine))
        print("Cosine similarity between gradients:", grad_cosine)
        sys.stdout = orig_stdout
        f.close()

            
        
        # plot cosine similarity per step #
        plt.plot(grad_cosine)
        plt.xlabel("Step")
        plt.ylabel("Cosine similarity")
        plt.title("Cosine similarity between gradients")
        plt.savefig( f"{tgt_save_path}/cosine_similarity.png")
        plt.show()
        plt.close()
        # write cosine similarity to file #
        with open(f"{tgt_save_path}/cosine_similarity.txt", "w+") as f:
            f.write("\n****\n")
            for item in grad_cosine:
                f.write("%s\n" % item)
        # write best accuracy to file #
        with open(result_path, "w+") as f:
                f.write("%s\n" % best_acc)
    
    
    
    
    


def main(args):
    args_update(args)
    print(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    model, preprocess = clip.load(args.backbone, device=args.device)

    domain_list = os.listdir(args.data_root)

    domain_list = [x for x in domain_list if ".txt" not in x]

    classnames_path = os.path.join(args.data_root, domain_list[0])

    classnames = os.listdir(classnames_path)
    n_cls = len(classnames)
    classnames.sort()

    args.output_dir = "outputs/" + str(args).replace(", ", "/").replace(
        "'", ""
    ).replace("(", "").replace(")", "").replace("Namespace", "")

    print("Output directory:", args.output_dir)
    # os.system("rm -rf " + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    args.n_cls = n_cls
    train(domain_list, classnames, model, preprocess, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Training and Evaluation Script", parents=[arg_parse()]
    )
    args = parser.parse_args()
    main(args)

import logging
import argparse
import yaml
import os
import time
import matplotlib.pyplot as plt
import random

import torch
from torch.optim import Adam, SGD

from pymoo.problems import get_problem

from zdt_functions import *

logging.getLogger("matplotlib.font_manager").disabled = True


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", default="zdt1",  type=str, help="Toy problem"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="The initial learning rate for Adam."
    )

    parser.add_argument(
        "--iters",
        default=20000,
        type=int,
        help="Total number of training steps to perform.",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")

    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = "outputs/" + str(args).replace(", ", "/").replace("'", "").replace(
        "(", ""
    ).replace(")", "").replace("Namespace", "")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    logging.basicConfig(
        filename=os.path.join(output_dir, "training.log"),
        level=logging.DEBUG,
    )

    x = torch.rand((1, 30))/200 +.5
    x.requires_grad = True
    optimizer = Adam([x], lr=args.lr)
    # optimizer = SGD([x], lr=args.lr)

    start_time = time.time()
    hv_results = []
    losses_1 = []
    losses_2 = []
    losses_1_1 = []
    losses_2_1 = []
    losses_1_2 = []
    losses_2_2 = []
    cosines = []
    for i in range(1+args.iters):
        loss_1, loss_2 = loss_function(x, problem=args.problem)
        pfront = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
        pfront = pfront.detach().cpu().numpy()

        loss_1.sum().backward(retain_graph=True)
        grad_1 = x.grad.detach().clone()
        x.grad.zero_()

        loss_2.sum().backward()
        grad_2 = x.grad.detach().clone()
        x.grad.zero_()

        # Perforam gradient normalization trick
        grad_1 = torch.nn.functional.normalize(grad_1, dim=0).detach().clone()
        grad_2 = torch.nn.functional.normalize(grad_2, dim=0).detach().clone()
        dx1 = 100*args.lr * grad_1
        dx2 = 100*args.lr * grad_2
        new_x_1 = x.data - dx1
        new_x_2 = x.data - dx2
        new_x_1.data = torch.clamp(new_x_1.data.clone(), min=1e-6, max=1.0 - 1e-6)
        new_x_2.data = torch.clamp(new_x_2.data.clone(), min=1e-6, max=1.0 - 1e-6)
        
        loss_1_1, loss_2_1 = loss_function(new_x_1, problem=args.problem)
        loss_1_2, loss_2_2 = loss_function(new_x_2, problem=args.problem)
        problem = get_problem(args.problem)
        x_p = problem.pareto_front()[:, 0]
        y_p = problem.pareto_front()[:, 1]
        plt.scatter(x_p, y_p, c="r")

        plt.scatter(loss_1.detach().cpu().numpy(), loss_2.detach().cpu().numpy(), c="black")
        # plt.scatter(loss_1_1.detach().cpu().numpy(), loss_2_1.detach().cpu().numpy(), c="b")
        # plt.scatter(loss_1_2.detach().cpu().numpy(), loss_2_2.detach().cpu().numpy(), c="g")
        
        # draw arrow from x, y to x+dx, y+dy
        plt.arrow(loss_1.item(), loss_2.item(), (loss_1_1-loss_1).item(), (loss_2_1-loss_2).item(), head_width = 0.02, ec="b")
        plt.arrow(loss_1.item(), loss_2.item(), (loss_1_2-loss_1).item(), (loss_2_2-loss_2).item(), head_width = 0.02, ec="g")
        plt.title(f"Iteration: {i}/{args.iters}")
        plt.tight_layout()
        plt.savefig("%s/%d.png" % (output_dir, i))
        plt.close()

        optimizer.zero_grad()

        grad = (grad_1 + grad_2)/2.
        cosine = torch.nn.functional.cosine_similarity(grad_1.reshape(-1), grad_2.reshape(-1), dim=0)
        cosines.append(cosine.item())
        x.grad = grad
        # x.data = x.data - args.lr * grad
        # if x.grad is not None:
            # x.grad.zero_()
        optimizer.step()

        x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.0 - 1e-6)

        log_str = f"Iteration: {i}/{args.iters}, Time: {time.time() - start_time:.2f}"
        print(log_str)
        logging.info(log_str)
        log_str = f"Loss_1: {loss_1.sum().detach().cpu().numpy()}, Loss_2: {loss_2.sum().detach().cpu().numpy()}"
        print(log_str)
        logging.info(log_str)
        log_str = f"Loss_1_1: {loss_1_1.sum().detach().cpu().numpy()}, Loss_2_1: {loss_2_1.sum().detach().cpu().numpy()}"
        print(log_str)
        logging.info(log_str)
        log_str = f"Loss_1_2: {loss_1_2.sum().detach().cpu().numpy()}, Loss_2_2: {loss_2_2.sum().detach().cpu().numpy()}"
        print(log_str)
        logging.info(log_str)
        losses_1.append(loss_1.item())
        losses_2.append(loss_2.item())
        losses_1_1.append(loss_1_1.item())
        losses_2_1.append(loss_2_1.item())
        losses_1_2.append(loss_1_2.item())
        losses_2_2.append(loss_2_2.item())
    logging.info(f"Losses_1:\n {losses_1}")
    logging.info(f"Losses_2:\n {losses_2}")
    logging.info(f"Losses_1_1:\n {losses_1_1}")
    logging.info(f"Losses_2_1:\n {losses_2_1}")
    logging.info(f"Losses_1_2:\n {losses_1_2}")
    logging.info(f"Losses_2_2:\n {losses_2_2}")
    logging.info(f"Cosines:\n {cosines}")
        
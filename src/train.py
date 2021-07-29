import os
import os.path as osp
import argparse
import logging
import random
import json
from tqdm import tqdm, trange
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from config import add_arguments
from data import WeiboDataset
from utils import make_input
from models.feat_mlp import FeatMLP
from models.feat_svm import train_svm
from models.resnet import ResNet
from models.text2labels import UserText2Labels, MicroblogText2Labels
from models.fusion_net import FusionNet
from models.pmfn import PMFN, pretrain_PMFN

models = {"FeatSVM": None, "FeatMLP": FeatMLP, "ResNet": ResNet, "UserText2Labels": UserText2Labels, "MicroblogText2Labels": MicroblogText2Labels, "FusionNet": FusionNet, "PMFN": PMFN}
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args, train_dataset, eval_dataset, model):
    sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iterator = tqdm(range(args.num_epochs), desc="Epoch")

    best_f1 = 0.0
    best_step = 0
    best_model = None
    patience = 0
    global_step = 0
    average_loss = 0

    logger.info("**** Running Training ****")
    logger.info("   Num epochs = %d", args.num_epochs)
    logger.info("   Num samples = %d", len(train_dataset))
    logger.info("   Batch size = %d", args.batch_size)
    logger.info("   Num steps = %d", args.num_epochs * len(train_dataset) // args.batch_size)

    for _ in iterator:
        for batch in dataloader:
            global_step += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = make_input(batch)
            logits, loss = model(**inputs)
            average_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % args.logging_steps == 0:
                logger.info("Average loss %s at global step %s", str(average_loss / args.logging_steps), str(global_step)) 
                average_loss = 0
            
            if global_step % args.eval_steps == 0:
                if args.eval_train:
                    logger.info("--- eval_train ---")
                    evaluate(args, train_dataset, model, "Train")
                eval_f1, eval_loss = evaluate(args, eval_dataset, model, "Eval")
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    best_step = global_step
                    best_model = deepcopy(model)
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        logger.info("Early Stopping!")
                        return best_model, best_step

                logger.info("best step at %d", best_step)
            
    return best_model, best_step

def evaluate(args, dataset, model, split="Train"):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    iterator = tqdm(dataloader, desc="Evaluating")

    eval_loss = 0.0
    eval_steps = 0
    preds = None
    labels = None

    logger.info("**** Running Evaluation ****")
    logger.info("   Num samples = %d", len(dataset))
    logger.info("   Batch size = %d", args.batch_size)

    for batch in iterator:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        y = batch[8]
        with torch.no_grad():
            inputs = make_input(batch)
            logits, loss = model(**inputs)
            eval_loss += loss.item()
        
        eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, y.detach().cpu().numpy(), axis=0)

    eval_loss /= eval_steps
    preds = np.argmax(preds, axis=1)
    # print(labels, preds)
    f1 = f1_score(labels, preds)
    logger.info("%s F1 = %s, Loss = %s", split, str(f1), str(eval_loss))
    if "precision" in args.eval_metric:
        logger.info("%s Precision = %s", split, precision_score(labels, preds))
    if "recall" in args.eval_metric:
        logger.info("%s Recall = %s", split, recall_score(labels, preds))
    if "accuracy" in args.eval_metric:
        logger.info("%s Accuracy = %s", split, accuracy_score(labels, preds))
    return f1, eval_loss

def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    config = json.load(open(osp.join("configs", args.config_file), "r"))
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("building dataset")
    if config["model"] in ["FusionNet", "UserText2Labels"]:
        user_only = True
        tokenizer = "xlnet" if config["emb_layer"] == "XLNet" else "word2vec"
    else:
        user_only = False
        if config["model"] == "PMFN":
            tokenizer = "xlnet" if config["text_emb_layer"] == "XLNet" else "word2vec"
        elif config["model"] == "MicroblogText2Labels":
            tokenizer = "xlnet" if config["emb_layer"] == "XLNet" else "word2vec"
        else:
            tokenizer = "word2vec"
    if config["model"] in ["PMFN", "ResNet"]:
        process_image = True
    else:
        process_image = False

    dataset = WeiboDataset(
        name=args.dataset,
        download_picture=args.download_picture,
        overwrite_cache=args.overwrite_cache,
        tokenizer=tokenizer,
        user_only=user_only,
        max_seq_len=args.max_seq_len,
        max_microblogs=args.max_microblogs,
        process_image=process_image
    )
    train_dataset, eval_dataset, test_dataset = random_split(dataset, args.split_size)

    if config["model"] in models:
        if config["model"] == "FeatSVM":
            train_svm(train_dataset, eval_dataset, test_dataset, args)
            return
        elif config["model"] == "FeatMLP":
            model = FeatMLP(args.num_features, config["hidden_size"], args.dropout)
        elif config["model"] == "ResNet":
            model = ResNet(args.dropout, args.max_microblogs)
        elif config["model"] == "UserText2Labels":
            model = UserText2Labels(args.dropout, config["hidden_size"], args.max_seq_len, config["emb_layer"], config["proc_layer"], config["num_layers"], config["attention"])
        elif config["model"] == "MicroblogText2Labels":
            hidden_sizes = {
                "text": config["hidden_size"],
                "classifier": config["hidden_size_classifier"],
                "squeezed": config["hidden_size_squeezed"],
            }
            model = MicroblogText2Labels(args.dropout, hidden_sizes, args.batch_size, args.max_seq_len, args.max_microblogs, config["emb_layer"], config["proc_layer"], config["num_emb_layers"], config["num_aggr_layers"])
        elif config["model"] == "FusionNet":
            hidden_sizes = {
                "text": config["hidden_size"],
                "task1": config["hidden_size_task1"],
                "task2": config["hidden_size_task2"]
            }
            model = FusionNet(args.num_features, hidden_sizes, args.dropout, config["emb_layer"], config["mid_layer"], config["attention"], args.max_seq_len, config["w1"], config["w2"])
        elif config["model"] == "PMFN":
            hidden_sizes = {
                "hidden_size": config["hidden_size"],
                "recon": config["hidden_size_recon_mlp"],
                "classifier": config["classifier_mlp"]
            }
            model = PMFN(args.num_features, hidden_sizes, args.dropout, config["text_emb_layer"], config["visual_emb_layer"], args.max_seq_len, args.max_microblogs, config["num_aggr_layers"], config["w1"], config["w2"])
        logger.info(model)
        model = model.to(args.device)
    else:
        logger.error("%s is not implemented!" % (config["model"]))
        raise NotImplementedError

    if config["model"] in ["PMFN"]:
        model = pretrain_PMFN(args, train_dataset, model)
    model, best_step = train(args, train_dataset, eval_dataset, model)
    test_f1, test_loss = evaluate(args, test_dataset, model, "Test")
    if args.save:
        logger.info("Saving Model")
        args.save_dir = osp.join(args.save_dir, "{}_{}".format(config["model"], args.dataset))
        if not osp.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(args, osp.join(args.save_dir, "training_args.txt"))
        torch.save(model.state_dict(), osp.join(args.save_dir, "model.pt"))
        with open(osp.join(args.save_dir, "result.txt"), "w") as f:
            f.write(str(args))
            f.write("best_step = {}, test_f1 = {}, test_loss = {}".format(best_step, test_f1, test_loss))

if __name__ == "__main__":
    main()
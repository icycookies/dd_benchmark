import argparse

def add_arguments(parser):
    # dataset parameters
    parser.add_argument("--dataset", type=str, default="weibo2012")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-microblogs", type=int, default=128)
    parser.add_argument("--download-picture", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--split-size", type=int, nargs="+", default=[1972, 247, 247])

    # basic training parameters
    parser.add_argument("--config-file", type=str, default="featmlp.json")
    parser.add_argument("--num-features", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=int)

    # pretraining parameters
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-gradient-accumulation-steps", type=int, default=1)

    # logging and saving parameters
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--eval-metric", type=str, nargs="+", default=["f1"])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", type=str, default="../checkpoints")

import torch
from lib.models.vit_dist import build_ostrack_dist
import argparse
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='vit_dist', help='Name of the train script.')
    parser.add_argument('--config', type=str, default='experiments/ostrack/ostrack_distillation_123_128_h128.yaml', help="Name of the config file.")
    args = parser.parse_args()

    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(args.config)
    model = build_ostrack_dist(cfg)
    ckpt = torch.load('/home/lpy/mae/output_dir/checkpoint-20.pth')
    ckpt = ckpt['model']
    a, b = model.load_state_dict(ckpt, strict=False)
    print(a)
    print(b)
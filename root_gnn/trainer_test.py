#!/usr/bin/env python 

from root_gnn.trainer import Trainer

def test():
    config_dir = '/global/cfs/cdirs/atlas/xju/software/root_gnn/configs/test_summary.yaml'
    trainer = Trainer(config_dir, distributed=False, verbose="INFO")
    trainer.execute()

if __name__ == "__main__":
    test()
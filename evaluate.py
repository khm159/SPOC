import hydra
import random
import torch
import numpy as np
import os

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    """
    Main function to run the evaluator.
    """
    # Set random seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize the evaluator with the configuration
    if 'spoc' in cfg.benchmark.name :
        from SPOC.eval.spoc_evaluator import SpocBenchEvaluator
        evaluator = SpocBenchEvaluator(cfg)
    else:
        NotImplementedError(f"Evaluator for benchmark {cfg.benchmark.name} is not implemented.")

    # Run the evaluation
    evaluator.evaluate()

if __name__=='__main__':
    main()
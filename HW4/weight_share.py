import argparse
import os
import warnings

import torch

from net.quantization import apply_weight_sharing
import util

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('--load-model-path', type=str,
                    default='saves/model_after_retraining.ptmodel', help='path to load pruned model')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--save-model-path', type=str,
                    default='saves/model_after_weight_sharing.ptmodel', help='path to save quantized model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

warnings.filterwarnings("ignore")


def main():
    # Define the model
    model = torch.load(args.load_model_path, weights_only=False)
    print('--- Before weight sharing ---')
    accuracy = util.test(model, use_cuda)
    util.log(f"Accuracy_before_quantization {accuracy}")

    # Weight sharing
    apply_weight_sharing(model)
    print('--- After weight sharing ---')
    accuracy = util.test(model, use_cuda)
    util.log(f"Accuracy_after_quantization {accuracy}")

    # Save the new model
    os.makedirs('saves', exist_ok=True)
    torch.save(model, args.save_model_path)


if __name__ == '__main__':
    main()


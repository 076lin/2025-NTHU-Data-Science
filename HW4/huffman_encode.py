import argparse

import torch

from net.models import AlexNet
from net.huffmancoding import huffman_encode_model, huffman_decode_model
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--load-model-path', type=str,
                    default='saves/model_after_weight_sharing.ptmodel', help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA')
parser.add_argument('--log', type=str,
                    default='log.txt', help='log file name')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')


def main():
    # Encode
    print(f"--- Load model ({args.load_model_path}) ---")
    model = torch.load(args.load_model_path, weights_only=False)
    print('--- Accuracy before encoding ---')
    accuracy = util.test(model, use_cuda)
    util.log(f"Accuracy_before_encoding {accuracy}", args.log)
    print(f"--- Start encoding ---")
    huffman_encode_model(model)

    # Decode
    print(f"--- Start decoding ---")
    model = AlexNet().to(device)
    huffman_decode_model(model)
    print(f"--- Accuracy after decoding ---")
    accuracy = util.test(model, use_cuda)
    util.log(f"Accuracy_after_decoding {accuracy}", args.log)


if __name__ == '__main__':
    main()






import os, sys
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for rmsnorm')
parser.add_argument('vocab_size', type=int, help='The vocab_size')
parser.add_argument('seq_len', type=int, help='The sequence length to use for rmsnorm')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for rmsnorm')
parser.add_argument('--output_file', default = 'llama-loss-output.bin', type=str, help='The output file to use for rmsnorm')

from transformers import AutoTokenizer, AutoModelForCausalLM
import fileio_utils

def generate_random_one_hot_matrix(num_rows, num_classes):
    
    indices = torch.randint(0, num_classes, (num_rows,), device=0)

    one_hot_matrix = torch.nn.functional.one_hot(indices, num_classes=num_classes).float()
    return one_hot_matrix


if __name__ == '__main__':
    compilation_error = os.system('make cel')
    compilation_error = os.system('make gd-to-AB')

    if compilation_error:
        print("Error compiling cel")
        exit(1)
    args = parser.parse_args()
    model_card = f"model-storage/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only = True, cache_dir = "./model-storage")
    layer = getattr(model.model.layers[0], f'input_layernorm')

    (embed_dim, ) = layer.weight.shape
    
    
    if not os.path.isfile(args.input_file):
        temp_X = torch.randn(args.seq_len, embed_dim, device = 0)
        fileio_utils.save_int(temp_X, 1 << 16, args.input_file)



    one_hot_file = "one_hot_matrix.bin"
    one_hot_matrix = generate_random_one_hot_matrix(args.vocab_size, args.seq_len)
    fileio_utils.save_int(one_hot_matrix, 1, one_hot_file)

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'

    # os.system(f'./cel {args.vocab_size} {one_hot_file} {args.input_file} {args.seq_len} {embed_dim} {workdir} {args.output_file}')

    os.system(f'./gd-to-AB {args.vocab_size} {one_hot_file} {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix}')


    # remove the rms_inv_temp.bin file

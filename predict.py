
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import argparse
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Model
from eval import Predict
from data_preprocess.data import get_npy

def get_tensor(fn, num_chunks, input_length):
    # load audio
    raw = get_npy(fn)

    if len(raw) < input_length:
        nnpy = np.zeros(input_length)
        ri = int(np.floor(np.random.random(1) * (input_length - len(raw))))
        nnpy[ri:ri+len(raw)] = raw
        raw = nnpy
    # split chunk
    length = len(raw)
    chunk_length = input_length
    hop = (length - chunk_length) // num_chunks
    x = torch.zeros(num_chunks, chunk_length)
    for i in range(num_chunks):
        x[i] = torch.Tensor(raw[i*hop:i*hop+chunk_length]).unsqueeze(0)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyper-parameters
    parser.add_argument('--conv_channels', type=int, default=128)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=513)
    parser.add_argument('--n_harmonic', type=int, default=6)
    parser.add_argument('--semitone_scale', type=int, default=2)
    parser.add_argument('--learn_bw', type=str, default='only_Q')

    # dataset
    parser.add_argument('--input_length', type=int, default=48000)
    parser.add_argument('--num_chunks', type=int, default=16, help='number of chunks to split the input audio')

    # training settings
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./pretrained')
    parser.add_argument('--model_load_path', type=str, default='./pretrained/best_model.pth')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=20)

    # input positional arguments
    parser.add_argument('input_path', type=str, help='input wav file path')

    config = parser.parse_args()
    # print(config)

    p = Predict(config)

    # load and split
    x = get_tensor(config.input_path, config.num_chunks, config.input_length)
    # forward
    prd = p.forward(x)

    # get probabilities
    estimated = prd.detach().numpy().astype(np.float32).mean(axis=0)

    # labels
    labels = ['Percussion', 'Bass', 'Chord', 'Melody', 'FX', 'Voice']

    # print winner class
    winner = np.argmax(estimated)
    print(f"\nWinner class: {labels[winner]} ({estimated[winner]:.4f})\n")

    # print confusion matrix sorted by probabilities
    sorted_indices = np.argsort(estimated)[::-1]
    print("Estimated probabilities:")
    for i, label in enumerate(labels):
        print(f"{estimated[sorted_indices[i]]:.4f}\t{label}")
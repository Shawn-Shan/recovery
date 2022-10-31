import sys

import os
import argparse
import torch, torchvision
import numpy as np
import tc
from tqdm import tqdm
import pickle

def gen_one_dist(gan_model):
    vector1 = np.random.uniform(0, 0.2, 512)
    vector2 = np.random.uniform(-0.5, 0.5, 512)
    resizer = torchvision.transforms.Resize((32, 32))

    num_images = 500
    batch_size = 7

    vector1 = np.array(vector1, dtype=np.float32)
    vector2 = np.array(vector2, dtype=np.float32)

    all_imgs = []

    for batch_i in tqdm(range(0, num_images, batch_size)):
        noise = torch.randn(batch_size, 512) * torch.tensor(vector1) + torch.tensor(vector2) + 0.1
        noise = tc.gpu(noise)

        with torch.no_grad():
            generated_images = gan_model.test(noise)
            generated_images = generated_images.clamp(min=-1, max=1) / 2 + 0.5

        generated_images = resizer(generated_images)
        generated_images = generated_images.permute(0, 2, 3, 1)
        generated_images = tc.cpu(generated_images)
        all_imgs.append(generated_images)

    all_imgs = np.concatenate(all_imgs)
    return all_imgs


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                         'PGAN', model_name='celebAHQ-512',
                         pretrained=True, useGPU=True)

    if not os.path.exists(args.hidden_dir):
        os.makedirs(args.hidden_dir)

    for j in range(args.total_number):
        print("Generate hidden dist {}...".format(j))
        cur_all_imgs = gen_one_dist(gan)
        pickle.dump(cur_all_imgs, open(os.path.join(args.hidden_dir, "{}.p".format(j)), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--total-number', type=str,
                        help='total number of hidden distribution to generate', default=100)
    parser.add_argument('--hidden-dir', type=str,
                        help='Result directory for the hidden dist', default='hidden')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()

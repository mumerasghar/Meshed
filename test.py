import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
from data import COCO, DataLoader, Flicker8k

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            images = images.to(device)
            with torch.no_grad():
                out= model(images, caps_gt).argmax(-1)
                print(out.shape)
            caps_gen = text_field.decode(out[0])
            print(f'generated captions are {caps_gen}')
            print(f'real captions are {caps_gt}')
            # for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
            #     gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            #     gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
            #     gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    cap_file = 'captions.pickle'
    dir_path = './Flicker8k_Dataset/'
    all_img_name = 'img_name.pickle'

    train_dataset = Flicker8k(dir_path, cap_file, all_img_name)
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(train_dataset.vocab_size, 54, 3, train_dataset.encoder.token_to_index['<pad>'])
    model = Transformer(train_dataset.encoder.token_to_index['<start>'], encoder, decoder).to(device)


    data = torch.load('./saved_models/m2_transformer_last.pth',map_location=torch.device('cpu'))
    model.load_state_dict(data['state_dict'])

    dict_dataloader_test = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, train_dataset.encoder)
    print(scores)

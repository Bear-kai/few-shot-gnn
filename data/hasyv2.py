from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
import random
from . import parser
from datasets.HASYv2.hasy_tools import load_images, generate_index
from tqdm import tqdm

class Hasyv2(data.Dataset):
    def __init__(self, root, dataset='hasyv2'):
        self.root = root
        self.seed = 10
        self.dataset = dataset
        self.kfold = 1
        self.task = 'classification-task'
        if not self._check_exists_():
            self._init_folders_()
            self._preprocess_()

    def _init_folders_(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))

    def _check_exists_(self):
        return os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_train.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_test.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_label_encoder.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_label_decoder.pickle')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_std.npy')) and \
               os.path.exists(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_mean.npy'))

    def _preprocess_(self):
        print('\nPreprocessing Hasyv2 images...')

        csv_train_filepath = os.path.join(self.root, 'HASYv2', self.task, 'fold-' + str(self.kfold) , 'train.csv')
        csv_test_filepath = os.path.join(self.root, 'HASYv2', self.task, 'fold-' + str(self.kfold), 'test.csv')

        train_symbol_id2index = generate_index(csv_train_filepath)
        test_symbol_id2index = generate_index(csv_test_filepath)

        # Unify indexes of train and test
        all_symbol_id = set(list(train_symbol_id2index.keys()) + list(test_symbol_id2index.keys()))
        # Label encoder symbol_id -> index
        label_encoder = {}
        # Label decoder index -> symbol_id
        label_decoder = {}
        for i,symbol_id in enumerate(all_symbol_id):
            label_encoder[symbol_id] = i
            label_decoder[i] = symbol_id

        train_data = load_images(csv_train_filepath, label_encoder, one_hot=False, flatten=False)
        test_data = load_images(csv_test_filepath, label_encoder, one_hot=False, flatten=False)

        train_data_tmp = [i[np.newaxis] for i in train_data[0]]
        train_data_tmp = np.vstack(train_data_tmp)
        train_data_mean = train_data_tmp.mean(0)
        train_data_std = train_data_tmp.std(0)

        # reformat the dataset as in imagenet
        train_set = {}
        for i in np.arange(len(train_data[0])):
            img = train_data[0][i]
            index = train_data[1][i]
            if index not in train_set:
                train_set[index] = []
            train_set[index].append(img)

        test_set = {}
        for i in np.arange(len(test_data[0])):
            img = test_data[0][i]
            index = test_data[1][i]
            if index not in test_set:
                test_set[index] = []
            test_set[index].append(img)

        with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_train.pickle'), 'wb') as handle:
            pickle.dump(train_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_test.pickle'), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)

        with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_label_decoder.pickle'), 'wb') as handle:
            pickle.dump(label_decoder, handle, protocol=2)

        np.save(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_mean.npy'), train_data_mean)
        np.save(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_std.npy'), train_data_std)

        print('Images preprocessed')

    def load_dataset(self, train, size):
        print("Loading dataset")
        if train:
            with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_train.pickle'), 'rb') as handle:
                data = pickle.load(handle)
        else:
            with open(os.path.join(self.root, 'compacted_datasets', 'hasyv2_test.pickle'), 'rb') as handle:
                data = pickle.load(handle)

        train_data_mean = np.load(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_mean.npy'))
        train_data_std = np.load(os.path.join(self.root, 'compacted_datasets', 'hasyv2_data_std.npy'))
        # remove the 3th channel dimension for the pil image loading.
        train_data_mean = train_data_mean[:,:,0]
        train_data_std = train_data_std[:, :, 0]

        name_rotated_data_file = 'hasyv2_test_rotated.pickle'
        if train:
            name_rotated_data_file = 'hasyv2_train_rotated.pickle'

        if os.path.exists(os.path.join(self.root, 'compacted_datasets', name_rotated_data_file)):
            with open(os.path.join(self.root, 'compacted_datasets', name_rotated_data_file), 'rb') as handle:
                data_rot = pickle.load(handle)
        else:
            print("Num classes before rotations: " + str(len(data)))
            data_rot = {}
            # resize images and normalize
            print("Rotating images from : %d classes from %s data ..." % (len(data),train))
            for class_ in tqdm(data):
                for rot in range(4):
                    data_rot[class_ * 4 + rot] = []
                    for i in range(len(data[class_])):
                        image2resize = pil_image.fromarray(np.uint8(data[class_][i][:,:,0]))
                        image_resized = image2resize.resize((size[1], size[0]))
                        # convert to float and normalize
                        image_resized = (np.array(image_resized, dtype='float32') - train_data_mean) / train_data_std
                        image = self.rotate_image(image_resized, rot)
                        image = np.expand_dims(image, axis=0)
                        data_rot[class_ * 4 + rot].append(image)

            # File bigger than 4GB we need protocol 4 of pickle
            with open(os.path.join(self.root, 'compacted_datasets', name_rotated_data_file), 'wb') as handle:
                pickle.dump(data_rot, handle, protocol=4)
            print("Num classes after rotations: " + str(len(data_rot)))

        print("Dataset Loaded")
        return data_rot

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[:, :] = np.rot90(image[:, :], k=times)
        return rotated_image
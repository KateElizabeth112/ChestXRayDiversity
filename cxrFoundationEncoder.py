import copy
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import numpy as np
import pickle as pkl
import os
import io 
import png

class CxrFoundationEncoder:
    def __init__(self, data, dataset_name):
        # check that the dataset parameter is an instance of Dataset
        assert isinstance(data, Dataset), "train_data is not an instance of Dataset"

        # check that the dataset_name parameter is a string
        assert isinstance(dataset_name, str), "dataset_name is not a string"

        self.data = data
        self.dataset_name = dataset_name
        
        # create a directory to store encodings
        if not os.path.exists("CxrFoundationEncodings"):
            os.mkdir("CxrFoundationEncodings")

        self.representations_dir = os.path.join("CxrFoundationEncodings", f"{self.dataset_name}")
        if not os.path.exists(self.representations_dir):
            os.mkdir(self.representations_dir)

        # set up a data loader. batch size must be 1 for SAMMed Encoder
        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)

    def encode(self, start_idx):
        # check the input parameters
        # check that the start_idx parameter is an integer
        assert isinstance(start_idx, int), "start_idx is not an integer"

        if 'elixrc_model' not in locals():
            elixrc_model = tf.saved_model.load('elixr-c-v2-pooled')
            elixrc_infer = elixrc_model.signatures['serving_default']

        if 'qformer_model' not in locals():
            qformer_model = tf.saved_model.load("pax-elixr-b-text")

        for k, (image, _) in enumerate(self.data_loader):
            if k >= start_idx:
                print(f"Encoding {self.dataset_name} image {k}")
                serialized_img_tf_example = self.np_to_tfexample(image.numpy().squeeze()).SerializeToString()

                elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
                elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

                # Step 2 - Invoke QFormer with Elixr-C embeddings
                # Initialize text inputs with zeros
                qformer_input = {
                    'image_feature': elixrc_embedding.tolist(),
                    'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
                    'paddings':np.zeros((1, 1, 128), dtype=np.float32).tolist(),
                }

                qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
                embedding_tf = qformer_output['all_contrastive_img_emb']
                embedding_np = embedding_tf.numpy()
                embedding_torch = torch.from_numpy(embedding_np).flatten().unsqueeze(0)

                # Save embedding
                f = open(os.path.join(self.representations_dir, "img_{}.pkl".format(k)), "wb")
                pkl.dump(embedding_torch, f)
                f.close()

    def retrieve(self, indices, encodings_dir):
        # check that we have a directory where the encodings are stored
        assert os.path.exists(encodings_dir), "Encodings directory does not exist"

        # retrieve pre-computed embeddings based on a list of indicies
        for p in range(indices.shape[0]):
            f = open(os.path.join(encodings_dir, "img_{}.pkl".format(indices[p])), "rb")
            embedding = pkl.load(f)
            f.close()

            if p == 0:
                vectors = copy.deepcopy(embedding)
            else:
                vectors = torch.cat((vectors, embedding), dim=0)

        return vectors.detach().numpy()
    
    def np_to_tfexample(self, image_array: np.ndarray) -> tf.train.Example:
        """Creates a tf.train.Example from a NumPy array."""
        # Convert the image to float32 and shift the minimum value to zero
        image = image_array.astype(np.float32)
        image -= image.min()

        if image_array.dtype == np.uint8:
            # For uint8 images, no rescaling is needed
            pixel_array = image.astype(np.uint8)
            bitdepth = 8
        else:
            # For other data types, scale image to use the full 16-bit range
            max_val = image.max()
            if max_val > 0:
                image *= 65535 / max_val  # Scale to 16-bit range
            pixel_array = image.astype(np.uint16)
            bitdepth = 16

        # Ensure the array is 2-D (grayscale image)
        if pixel_array.ndim != 2:
            raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

        # Encode the array as a PNG image
        output = io.BytesIO()
        png.Writer(
            width=pixel_array.shape[1],
            height=pixel_array.shape[0],
            greyscale=True,
            bitdepth=bitdepth
        ).write(output, pixel_array.tolist())
        png_bytes = output.getvalue()

        # Create a tf.train.Example and assign the features
        example = tf.train.Example()
        features = example.features.feature
        features['image/encoded'].bytes_list.value.append(png_bytes)
        features['image/format'].bytes_list.value.append(b'png')

        return example

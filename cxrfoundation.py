# @title Fetch Sample Image
from PIL import Image
from IPython.display import Image as IPImage, display
import subprocess
import io
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import numpy as np
import png

from huggingface_hub import login
login()


# Helper function for processing image data
def np_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
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

#display(IPImage(filename="Chest_Xray_PA_3-8-2010.png", height=300))
#img = Image.open("Chest_Xray_PA_3-8-2010.png").convert('L')  # Convert to grayscale

# @title Invoke Model with Image
import numpy as np
import matplotlib.pyplot as plt

# Download the model repository files
#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="google/cxr-foundation",local_dir='.',
#                  allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])

# set up the dataloader
import torchvision.transforms as transforms
from cheXpertDataset import CheXpertDataset
import os
import torch

root_dir = '/Users/katephd/Documents/data'
dataset = CheXpertDataset(os.path.join(root_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

if 'elixrc_model' not in locals():
    elixrc_model = tf.saved_model.load('elixr-c-v2-pooled')
    elixrc_infer = elixrc_model.signatures['serving_default']

if 'qformer_model' not in locals():
    qformer_model = tf.saved_model.load("pax-elixr-b-text")


for i, (img, _) in enumerate(data_loader):
    # Step 1 - ELIXR C (image to elixr C embeddings)
    serialized_img_tf_example = np_to_tfexample(img.numpy().squeeze()).SerializeToString()

    elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
    elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

    print("ELIXR-C - interim embedding shape: ", elixrc_embedding.shape)

    # Step 2 - Invoke QFormer with Elixr-C embeddings
    # Initialize text inputs with zeros
    qformer_input = {
        'image_feature': elixrc_embedding.tolist(),
        'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
        'paddings':np.zeros((1, 1, 128), dtype=np.float32).tolist(),
    }

    qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
    elixrb_embeddings = qformer_output['all_contrastive_img_emb']

    print("ELIXR-B - embedding shape: ", elixrb_embeddings.shape)

    # Plot output
    plt.imshow(elixrb_embeddings[0], cmap='gray')
    plt.colorbar()  # Show a colorbar to understand the value distribution
    plt.title('Visualization of ELIXR-B embedding output')
    plt.show()

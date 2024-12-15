
import os

from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix


def main():
    '''AUGMENTATION SECTION'''

    augment_dir = "./inputImg"
    fractal_dir = "./fractalDemo"

    # Load the dataset (x_train) for augmentation
    train_dataset = datasets.ImageFolder(root=augment_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(fractal_dir)

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    #Prompts
    prompts = ["slightly bright"]

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class=idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    # Directory to save the augmented images
    aug_dir = "./augmented_images"
    os.makedirs(aug_dir, exist_ok=True)

    for idx, (image, label) in enumerate(augmented_train_dataset):
        image.save(f'augmented_images/{idx}.png')
        #print(f'Image index: {idx}, Label: {label}')
        pass

    '''AUGMENTATION ENDS'''



if __name__ == '__main__':
    main()


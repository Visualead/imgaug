import os
import os.path as osp

import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm


def prepare_sequential_augmentations():
    """
    Controls which sequential augmentations to perform

    For more examples, see: https://github.com/aleju/imgaug
    :return: list of different sequential augmentation instances
    """
    flips = [iaa.Fliplr(0, name='flip_0'), iaa.Fliplr(1, name='flip_1')]
    scales = [iaa.Affine(scale={"x": (0.5), "y": (0.5)}, name='scale_0.5'),
              iaa.Noop(name='scale_1'),
              iaa.Affine(scale={"x": (1.5), "y": (1.5)}, name='scale_1.5')]
    scales = [iaa.Rescale(scale=0.5, name='scale_0.5'),
              iaa.Noop(name='scale_1'),
              iaa.Rescale(scale=1.5, name='scale_1.5')]
    scales = [iaa.Resize(smaller_dim_size=240, name='scale_240'),
              iaa.Resize(smaller_dim_size=480, name='scale_480'),
              iaa.Resize(smaller_dim_size=720, name='scale_720')]


    rotations = [iaa.Affine(rotate=(-10), name='rot_350'),
                 iaa.Noop(name='rotate_0'),
                 iaa.Affine(rotate=(+10), name='rot_10')]
    rotations = [iaa.Rotate(angle=(-10), name='rot_350'),
                 iaa.Noop(name='rotate_0'),
                 iaa.Rotate(angle=(+10), name='rot_10')]


    sequential_augs = []
    for flip in flips:
        for scale in scales:
            for rotation in rotations:
                name = '_'.join([flip.name, scale.name, rotation.name])
                sequential_augs.append(
                    iaa.Sequential([flip, scale, rotation], name=name))

    return sequential_augs


def prepare_image_paths(db_dir):
    """
    Returns all the full file paths of the input directory

    :param db_dir: the database to be os.walked on
    :return:
    """
    image_paths = []
    for root, dirs, files in os.walk(db_dir):
        for file in files:
            if file[0] == '.':
                continue  # ignore hidden files
            image_paths.append(osp.join(root, file))
    return image_paths


def augment_database(db_dir, aug_dest_dir):
    """
    Performs various augmentations on an image database

    Runtime: ~5 sec per image for 18 different augmentations on a macbook pro 2015

    :param db_dir: the database directory to augment
    :param aug_dest_dir: the augmented database destination directory
    :return:
    """
    seq_augs = prepare_sequential_augmentations()

    image_paths = prepare_image_paths(db_dir=db_dir)
    relative_image_paths = [path[len(db_dir) + 1:] for path in image_paths]

    for image_path, rel_image_path in tqdm(
            zip(image_paths, relative_image_paths)):
        for seq_aug in tqdm(seq_augs):
            current_aug_dir = osp.join(aug_dest_dir, seq_aug.name)
            # todo: work in batches? -- probably faster
            image = cv2.imread(image_path)
            image_aug = seq_aug.augment_images([image])

            # save image to the correct relative path in this augmentation
            save_path = osp.join(current_aug_dir, rel_image_path)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path, image_aug[0])


if __name__ == '__main__':
    db_dir = '/Users/eddie/Documents/Temp/TestLossSet/Frames'
    aug_dest_dir = '/Users/eddie/Documents/Temp/TestLossSet/Augmentation5'

    augment_database(db_dir, aug_dest_dir)

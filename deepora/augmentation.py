"""MONAI image/mask transforms retained from the training pipeline."""

from monai.transforms import (
    Compose,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
)


def get_3D_image_augmentator(config, with_seg=False):
    all_keys = ["image", "mask"] if with_seg else ["image"]
    aug_ls = []
    if "RandZoom" in config:
        if with_seg:
            aug_ls.append(
                RandZoomd(
                    keys=all_keys,
                    mode=["bilinear", "nearest"],
                    min_zoom=config["RandZoom"]["min_zoom"],
                    max_zoom=config["RandZoom"]["max_zoom"],
                    prob=config["RandZoom"]["prob"],
                )
            )
        else:
            aug_ls.append(
                RandZoomd(
                    keys=["image"],
                    mode="bilinear",
                    min_zoom=config["RandZoom"]["min_zoom"],
                    max_zoom=config["RandZoom"]["max_zoom"],
                    prob=config["RandZoom"]["prob"],
                )
            )
    if "RandFlip" in config:
        aug_ls.append(
            RandFlipd(
                keys=all_keys,
                spatial_axis=config["RandFlip"]["spatial_axis"],
                prob=config["RandFlip"]["prob"],
            )
        )
    if "RandRotate90" in config:
        aug_ls.append(
            RandRotate90d(
                keys=all_keys,
                prob=config["RandRotate90"]["prob"],
                max_k=config["RandRotate90"]["max_k"],
                spatial_axes=config["RandRotate90"]["spatial_axes"],
            )
        )
    if "RandAffine" in config:
        if with_seg:
            aug_ls.append(
                RandAffined(
                    keys=all_keys,
                    mode=["bilinear", "nearest"],
                    prob=config["RandAffine"]["prob"],
                    rotate_range=config["RandAffine"]["rotate_range"],
                    translate_range=config["RandAffine"]["translate_range"],
                    padding_mode="zeros",
                )
            )
        else:
            aug_ls.append(
                RandAffined(
                    keys=["image"],
                    mode="bilinear",
                    prob=config["RandAffine"]["prob"],
                    rotate_range=config["RandAffine"]["rotate_range"],
                    translate_range=config["RandAffine"]["translate_range"],
                    padding_mode="zeros",
                )
            )
    if "RandGaussianNoise" in config:
        aug_ls.append(
            RandGaussianNoised(
                keys=["image"],
                prob=config["RandGaussianNoise"]["prob"],
                mean=config["RandGaussianNoise"]["mean"],
                std=0.1,
            )
        )
    if "RandGaussianSharpen" in config:
        aug_ls.append(
            RandGaussianSharpend(
                keys=["image"], prob=config["RandGaussianSharpen"]["prob"]
            )
        )
    if "RandScaleIntensity" in config:
        aug_ls.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=config["RandScaleIntensity"]["factors"],
                prob=config["RandScaleIntensity"]["prob"],
            )
        )
    if "RandAdjustContrast" in config:
        aug_ls.append(
            RandAdjustContrastd(
                keys=["image"],
                gamma=config["RandAdjustContrast"]["gamma"],
                prob=config["RandAdjustContrast"]["prob"],
            )
        )
    if len(aug_ls) > 0:
        augmentator = Compose(aug_ls)
    else:
        augmentator = None
    return augmentator

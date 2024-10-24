from datasets import Dataset, DatasetDict, Image
import os
from typing import List  


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset


def create_train_val_test_dataset(img_path: str | os.PathLike, 
                                  label_path: str | os.PathLike, 
                                  train_list: List[str] = None, 
                                  val_list: List[str] = None, 
                                  test_list: List[str] = None, 
                                  img_file_type: str = '.jpg', 
                                  label_file_type: str = '.png'):
    '''
    # your images can of course have a different extension
    # semantic segmentation maps are typically stored in the png format
    '''
    datasets = dict()
    for name_sample, sample in zip(['train', 'validation', 'test'], [train_list, val_list, test_list]):
        if sample:
            image_paths = list(map(lambda x: os.path.join(img_path, x+img_file_type if img_file_type else x), sample))
            print(name_sample, len(image_paths))
            label_paths = list(map(lambda x: os.path.join(label_path, x+label_file_type if label_file_type else x), sample))
            datasets[name_sample] = create_dataset(image_paths, label_paths)
        else:
            print(f'[INFO] Datasets is create without {name_sample} sample.')
    
    dataset_dict = DatasetDict(datasets)
    
    return dataset_dict
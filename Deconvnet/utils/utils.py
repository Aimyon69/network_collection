import torch
from typing import List
import os
import requests
import json

def decode_predictions(preds: torch.Tensor,top: int = 5) -> List[tuple]:
    class_index_path = ('https://s3.amazonaws.com'
                    '/deep-learning-models/image-models/imagenet_class_index.json')
    
    class_index_dict = None

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predciton'
        '(i.e. a 2D array of shape (samples, 1000)).'
        'Found array with shape: ' + str(preds.shape)
        )
    
    os.makedirs('./data',exist_ok=True)
    if not os.path.exists('./data/imagenet_class_index.json') :
        r = requests.get(class_index_path)
        with open('./data/imagenet_class_index.json','wb') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)

    results: List[tuple] = []

    for pred in preds:
        top_value,top_indices = torch.topk(pred,top)
        result = [tuple(class_index_dict[str(i.item())]) + (j.item(),) for (i,j) in zip(top_indices,top_value)]
        results.append(result)

    return results
    
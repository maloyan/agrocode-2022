# import clip
# from clip.clip import _MODELS, _download
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm.auto import tqdm

# from transformers import CLIPProcessor, CLIPVisionModel
device = "cuda"

queries = pd.read_csv("data/queries.csv")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# clip_code = "ViT-L/14@336px"
# model_path = os.path.expanduser("~/.cache/clip/ViT-L-14-336px.pt")
# with open(model_path, "rb") as opened_file:
#     clip_vit = torch.jit.load(opened_file, map_location="cuda:0").visual.eval()


class BlendModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )
        # self.avg = nn.AdaptiveAvgPool1d(256)

    def forward(self, x):
        x1 = transforms.functional.resize(x, size=[224, 224]) / 255.0
        x1 = transforms.functional.normalize(
            x1,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        x2 = transforms.functional.resize(x, size=[256, 256]) / 255.0
        x2 = transforms.functional.normalize(
            x2,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        # x1 = self.feature_extractor(x)
        # x1 = self.avg(x1.flip([-2]))
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)

        return (x1 + x2) / 2
        #x2 = self.avg(x2.flip([-2])) * 0.5 + self.avg(x2) * 0.5
        # x2 = self.avg(x2)
        # return x2 # #(x1 * 1.1) + (x2 * 0.9)  # Idea of Gong zhen

def extract_features(path, model):
    with torch.no_grad():
        image = (
            torch.Tensor(np.moveaxis(np.array(Image.open(path).convert("RGB")), -1, 0))
            .unsqueeze(0)
            .to(device)
        )
        feature = model(image).squeeze(0).detach().cpu().numpy()
        return feature / np.linalg.norm(feature)

def get_scores(model_name):
    model = BlendModel(model_name=model_name)
    model.to(device)
    model.eval()

    N = 10

    test_embeddings = np.array([extract_features(f"data/test/{idx}.png", model) for idx in tqdm(test.idx)])
    queries_embeddings = np.array([extract_features(f"data/queries/{idx}.png", model) for idx in tqdm(queries.idx)])

    neigh = NearestNeighbors(n_neighbors=N, metric="cosine")
    neigh.fit(test_embeddings)

    distances, idxs = neigh.kneighbors(queries_embeddings, N, return_distance=True)

    pred_data = pd.DataFrame()
    pred_data["score"] = 1 - distances.flatten()
    pred_data["database_idx"] = [test.idx.iloc[x] for x in idxs.flatten()]
    pred_data.loc[:, "query_idx"] = np.repeat(queries.idx, N).values
    return pred_data


# model = CLIPVisionModel.from_pretrained("clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")

pred1 = get_scores("convnext_xlarge_in22ft1k")
#pred2 = get_scores("convnext_xlarge_384_in22ft1k")

#res = pred1.merge(pred2, on=['database_idx', 'query_idx'])

#res["score"] = res["score_x"] + res["score_y"]
#pred_data = res[['database_idx', 'query_idx', 'score']]
pred1.to_csv("data/submission.csv", index=False)

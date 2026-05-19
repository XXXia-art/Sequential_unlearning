import os, sys, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import torch_fidelity
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class Generate_Dataset(Dataset):
    def __init__(self, path, content, sub_root, override_text=None):
        super().__init__()
        root_path = os.path.join(path, content, sub_root)
        self.content = content

        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"Missing image directory: {root_path}")

        self.images = [os.path.join(root_path, name) for name in os.listdir(root_path)]
        if not self.images:
            raise ValueError(f"No images found in: {root_path}")

        if content == 'coco':
            df = pd.read_csv("data/mscoco.csv")
            base_texts = [
                df.loc[
                    df['image_id'].isin([
                        int(os.path.basename(x).replace('COCO_val2014_', '').split('.')[0])
                    ]),
                    'text'
                ].tolist()[0]
                for x in self.images
            ]
        else:
            base_texts = [('_').join(os.path.basename(x).split('_')[:-1]) for x in self.images]

        if override_text is not None and content != 'coco':
            # 将完整 prompt 中的原 concept 替换成新的 cs_target_contents
            self.texts = [text.replace(content, override_text) for text in base_texts]
        else:
            self.texts = base_texts

    def __len__(self,):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'image': self.images[idx], 'content': self.content}


class CLIP_Score():
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda' if torch.cuda.is_available() else 'cpu'):
        local_model_path = "/data/coding/model_weight/CLIP/models--openai--clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(local_model_path, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(local_model_path, local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(local_model_path, local_files_only=True)
        self.device = device
        self.model = self.model.to(self.device)
    
    def __call__(self, dataloader):
        out_score = 0
        for item in dataloader:
            out_score_matrix  = self.model_output(images=item['image'], texts=item['text'])
            out_score += out_score_matrix.mean().item() 
        return out_score / len(dataloader)
    
    def model_output(self, images, texts):
        torch.cuda.empty_cache()
        images_feats = self.processor(images=[Image.open(img) for img in images], return_tensors="pt").to('cuda')
        images_feats = self.model.get_image_features(**images_feats)

        texts_feats = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt",).to('cuda')
        texts_feats = self.model.get_text_features(**texts_feats)

        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        score = (images_feats * texts_feats).sum(-1)
        return score


def find_root_paths(root_dir, sub_root):
    return sorted(
        list({os.path.abspath(os.path.join(dirpath, '..')) 
                for dirpath, dirnames, _ in os.walk(root_dir) if sub_root in dirnames})
    )


def split_contents_arg(raw_text):
    return [item.strip() for item in raw_text.split(';') if item.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--fid_target_contents', type=str, default=None)
    parser.add_argument('--cs_target_contents', type=str, default=None)
    parser.add_argument('--sub_root', type=str, default='edit')
    parser.add_argument('--pretrained_path', type=str)
    args = parser.parse_args()

    contents = split_contents_arg(args.contents)
    fid_target_contents = (split_contents_arg(args.fid_target_contents)
        if args.fid_target_contents is not None
        else contents
    )
    cs_target_contents = (
    split_contents_arg(args.cs_target_contents)
    if args.cs_target_contents is not None
    else contents
    )

    assert len(contents) == len(fid_target_contents), \
        "contents 和 fid_target_contents 的长度必须一致"
    assert len(contents) == len(cs_target_contents), \
    "contents 和 cs_target_contents 的长度必须一致"

    root_paths = find_root_paths(args.root_path, args.sub_root)
    print(f"Caculating CLIP Score and FID for {root_paths}... \n")
    CS_calculator = CLIP_Score()

    for root_path in root_paths:
        try:
            save_txt = os.path.join(root_path, 'record_metrics.txt')
            if not os.path.exists(save_txt): 
                with open(save_txt, 'a') as f:
                    f.writelines('*************************** \n')
                    f.writelines(f'Calculating the metrics for {root_path} \n')

            with open(save_txt, 'r') as f:  
                txt_content = f.read()
            for content, fid_target, cs_target in tqdm(list(zip(contents, fid_target_contents, cs_target_contents))):
                metric_tag = f"{content} | FID={fid_target} | CS={cs_target}"
                if metric_tag in txt_content:
                    continue
                dataset = Generate_Dataset(
                    root_path,
                    content,
                    args.sub_root,
                    override_text=cs_target
                )
                dataloader = DataLoader(dataset, batch_size=10)
                CS = CS_calculator(dataloader)
                fid_ref_path = (
                    os.path.join(args.pretrained_path, fid_target, 'original')
                    if fid_target != 'coco'
                    else "data/pretrain/coco/coco/original"
                )

                try:
                    FIDELITY = torch_fidelity.calculate_metrics(
                        input1=os.path.join(root_path, content, args.sub_root),
                        input2=fid_ref_path,
                        cuda=True,
                        fid=True,
                        verbose=False,
                    )
                    fid_val = FIDELITY['frechet_inception_distance']
                except Exception as fid_err:
                    print(f"[WARN] FID failed for {content}, skipping: {fid_err}")
                    fid_val = -1
                with open(save_txt, 'a') as f:
                    f.writelines(
                        f"{metric_tag}: CS is {CS * 100}, FID is {fid_val} \n"
                    )
                print(f"{metric_tag}: CS is {CS * 100}, FID is {fid_val}")

        except Exception as e:
            save_txt = os.path.join(root_path, 'record_metrics.txt')
            with open(save_txt, 'a') as f:
                f.writelines(f"[ERROR] Failed to score {root_path}: {e}\n")
            print(f"[ERROR] Failed to score {root_path}: {e}", file=sys.stderr)

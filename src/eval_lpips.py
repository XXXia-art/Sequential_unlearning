import os, sys, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import lpips

from PIL import Image
from tqdm import tqdm


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float()
    image = image / 127.5 - 1.0
    return image.unsqueeze(0)


def find_root_paths(root_dir, sub_root):
    return sorted(
        list({os.path.abspath(os.path.join(dirpath, '..'))
                for dirpath, dirnames, _ in os.walk(root_dir) if sub_root in dirnames})
    )


def split_contents_arg(raw_text):
    return [item.strip() for item in raw_text.split(';') if item.strip()]


def get_png_pairs(edit_dir, ref_dir):
    edit_files = sorted([x for x in os.listdir(edit_dir) if x.endswith(".png")])
    ref_files = sorted([x for x in os.listdir(ref_dir) if x.endswith(".png")])
    if len(edit_files) == 0 or len(ref_files) == 0:
        raise ValueError(f"No png files found in {edit_dir} or {ref_dir}")
    if edit_files != ref_files:
        raise ValueError(f"png filenames are not aligned between {edit_dir} and {ref_dir}")
    return [(os.path.join(edit_dir, x), os.path.join(ref_dir, x)) for x in edit_files]


class LPIPS_Score():
    def __init__(self, net='alex', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, img_pairs):
        out_score = 0
        for img1_path, img2_path in img_pairs:
            img1 = load_image(img1_path).to(self.device)
            img2 = load_image(img2_path).to(self.device)
            if img1.shape[-2:] != img2.shape[-2:]:
                img2 = torch.nn.functional.interpolate(
                    img2, size=img1.shape[-2:], mode='bilinear', align_corners=False
                )
            score = self.model(img1, img2)
            out_score += score.item()
        return out_score / len(img_pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--sub_root', type=str, default='edit')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--net', type=str, default='alex')
    args = parser.parse_args()

    contents = split_contents_arg(args.contents)

    root_paths = find_root_paths(args.root_path, args.sub_root)
    print(f"Caculating LPIPS for {root_paths}... \n")
    LPIPS_calculator = LPIPS_Score(net=args.net)

    for root_path in root_paths:
        try:
            save_txt = os.path.join(root_path, 'record_lpips.txt')
            if not os.path.exists(save_txt):
                with open(save_txt, 'a') as f:
                    f.writelines('*************************** \n')
                    f.writelines(f'Calculating LPIPS for {root_path} \n')

            with open(save_txt, 'r') as f:
                txt_content = f.read()

            for content in tqdm(contents):
                metric_tag = f"{content} | LPIPS={content}"
                if metric_tag in txt_content:
                    continue

                edit_dir = os.path.join(root_path, content, args.sub_root)
                if args.pretrained_path is None:
                    ref_dir = os.path.join(root_path, content, 'original')
                else:
                    ref_dir = os.path.join(args.pretrained_path, content, 'original')

                img_pairs = get_png_pairs(edit_dir, ref_dir)
                score = LPIPS_calculator(img_pairs)

                with open(save_txt, 'a') as f:
                    f.writelines(f"{metric_tag}: LPIPS is {score} \n")
                print(f"{metric_tag}: LPIPS is {score}")

        except Exception as e:
            save_txt = os.path.join(root_path, 'record_lpips.txt')
            with open(save_txt, 'a') as f:
                f.writelines(f"[ERROR] Failed to score {root_path}: {e}\n")
            print(f"[ERROR] Failed to score {root_path}: {e}", file=sys.stderr)

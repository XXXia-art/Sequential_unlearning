#!/usr/bin/env python3
import os, re, time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline

from src.utils import seed_everything


def get_token_id(prompt, tokenizer, return_ids_only=True):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokens.input_ids if return_ids_only else tokens


@torch.no_grad()
def analyze_edit_direction(delta, delta_history, k_e, t_vec, a_vec, P, P_hist, name, step, config_path):
    """
    分析当前编辑方向与历史/目标/保留的关系
    """
    log_lines = []
    
    # 1. 当前编辑在历史主方向上的分量 vs 正交分量
    if delta_history.norm() > 1e-8:
        U_h, S_h, _ = torch.linalg.svd(delta_history, full_matrices=False)
        top_k = min(5, U_h.shape[1])
        U_h_top = U_h[:, :top_k]
        
        delta_parallel = U_h_top @ U_h_top.T @ delta
        delta_orth = delta - delta_parallel
        
        parallel_ratio = (delta_parallel.norm() / delta.norm()).item() if delta.norm() > 1e-8 else 0
        orth_ratio = (delta_orth.norm() / delta.norm()).item() if delta.norm() > 1e-8 else 0
    else:
        parallel_ratio = 0
        orth_ratio = 1.0
    
    # 2. 当前编辑在目标概念方向上的能量
    delta_t_proj = (delta @ t_vec.T).norm().item() if t_vec.norm() > 1e-8 else 0
    delta_a_proj = (delta @ a_vec.T).norm().item() if a_vec.norm() > 1e-8 else 0
    
    # 3. P_hist 对当前编辑的压缩程度
    compression_ratio = ((P_hist @ delta).norm() / delta.norm()).item() if delta.norm() > 1e-8 else 1.0
    
    # 4. 保留空间维度信息
    retain_dims = P.shape[1] if len(P.shape) > 1 else P.shape[0]
    
    # 5. 编辑效率 = 目标方向投影 / 编辑总幅度
    edit_efficiency = delta_t_proj / (delta.norm().item() + 1e-8)
    
    log_lines.append(
        f"    [EditDir] parallel_hist={parallel_ratio:.4f} | "
        f"orth_hist={orth_ratio:.4f} | t_proj={delta_t_proj:.4f} | "
        f"a_proj={delta_a_proj:.4f} | compress={compression_ratio:.4f} | "
        f"efficiency={edit_efficiency:.4f}\n"
    )
    
    with open(config_path, "a") as f:
        f.writelines(log_lines)
    
    return {
        'parallel_ratio': parallel_ratio,
        'orth_ratio': orth_ratio,
        'delta_t_proj': delta_t_proj,
        'delta_a_proj': delta_a_proj,
        'compression_ratio': compression_ratio,
        'edit_efficiency': edit_efficiency
    }


@torch.no_grad()
def edit_model(
    args,
    pipeline,
    base_state,
    target_concepts,
    anchor_concepts,
    retain_texts,
    m_hist,
    v_hist,
    emb_size=768,
    chunk_size=128,
    device="cuda"
):
    I = torch.eye(emb_size, device=device)

    ## choose params
    if args.params == "V":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_v" in k }
    elif args.params == "K":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_k" in k }
    elif args.params == "KV":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_k" in k or "attn2.to_v" in k}
    else:
        raise ValueError("Invalid --params")

    # ---- null cluster ----
    null_inputs = get_token_id("", pipeline.tokenizer, return_ids_only=False)
    null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0]
    _, centers = kmeans(X=null_hidden[1:], num_clusters=3, device=device)
    K2 = torch.cat([null_hidden[[0]], centers.to(device)], dim=0).T
    I2 = torch.eye(K2.shape[1], device=device)
    
    ## Target / Anchor
    sum_tt, sum_at, ke = [], [], []
    t_vecs = []  # 保存用于分析
    a_vecs = []  # 保存用于分析

    for t, a in zip(target_concepts, anchor_concepts):
        t_in = get_token_id(t, pipeline.tokenizer, return_ids_only=False)
        a_in = get_token_id(a, pipeline.tokenizer, return_ids_only=False)

        t_emb = pipeline.text_encoder(t_in.input_ids.to(device)).last_hidden_state[0]
        a_emb = pipeline.text_encoder(a_in.input_ids.to(device)).last_hidden_state[0]

        idx_t = t_in.attention_mask[0].sum().item() - 2
        idx_a = a_in.attention_mask[0].sum().item() - 2

        t_vec = t_emb[[idx_t]]
        a_vec = a_emb[[idx_a]]

        sum_tt.append(t_vec.T @ t_vec)
        sum_at.append(a_vec.T @ t_vec)
        ke.append(t_vec.T)
        t_vecs.append(t_vec)
        a_vecs.append(a_vec)
        
    sum_tt = torch.stack(sum_tt).mean(0)
    sum_at = torch.stack(sum_at).mean(0)
    k_e = torch.stack(ke).mean(0)
    t_vec_avg = torch.stack(t_vecs).mean(0)  # 平均目标向量
    a_vec_avg = torch.stack(a_vecs).mean(0)  # 平均锚点向量

    ## Retain
    last_ret_embs = []

    for i in range(0, len(retain_texts), chunk_size):
        r_in = get_token_id(retain_texts[i:i + chunk_size], pipeline.tokenizer, return_ids_only=False)
        r_emb = pipeline.text_encoder(r_in.input_ids.to(device)).last_hidden_state
        idx = r_in.attention_mask.sum(1) - 2
        last_ret_embs.append(r_emb[torch.arange(r_emb.size(0)), idx].unsqueeze(1))

    last_ret_embs = torch.cat(last_ret_embs, dim=0)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]

    ## 记录指标
    delta_coef = args.delta_coef
    eta = args.eta
    match = re.search(r"step_(\d+)", args.save_path)
    step = int(match.group(1))
    config_path = os.path.join(os.path.dirname(os.path.dirname(args.save_path)), "config.txt")
    with open(config_path, "a") as f:
        f.write(
            f"[step {step:03d}] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n"
            f"current_targets: {', '.join(target_concepts)}\n"
        )

    global_hist_norm_sq = 0.0
    global_cur_norm_sq = 0.0

    ## Edit each layer
    for name, W in tqdm(edit_dict.items(), desc="Editing"):

        W = W.to(device)
        W_orig = base_state[name].to(device)
        delta_history = W - W_orig
        resp = delta_history @ k_e          
        noise = resp.norm(dim=0).pow(2)
        cache = torch.cat([resp, delta_history], dim=1)

        if name not in m_hist:
            m_hist[name] = noise
        if name not in v_hist:
            v_hist[name] = 0.0

        m_prev = float(m_hist[name])
        v_prev = float(v_hist[name])

        # 再判断是否触发 DeltaEdit
        std = v_prev ** 0.5
        trigger_deltaedit = (step > 5) and std != 0 and noise > eta * std + m_prev
        
        with open(config_path, "a") as f:
            f.write(
                f"layer={name}| \n"
                f"noise={float(noise):.8f} | "
                f"m_prev={float(m_prev):.8f} | "
                f"std={float(std):.8f} | \n "
            )

        # ========== DeltaEdit: 构建 P_hist ==========
        if trigger_deltaedit:
            print(f"[Deltaedit] ---------------")
            with open(config_path, "a") as f:
                f.write(f"    >>> DeltaEdit TRIGGERED at step {step:03d}, layer={name}\n")

            D_hist = cache @ cache.T
            U_hist, S_hist, _ = torch.linalg.svd(D_hist, full_matrices=False)
            # 只投影掉 top1 方向，给当前编辑留出更多空间
            if S_hist.numel() > 0 and S_hist[0] > 0.1:
                U_sel = U_hist[:, [0]]  # 只取第1个主方向
                P_hist = torch.eye(W.shape[0], device=device, dtype=W.dtype) - U_sel @ U_sel.T
                
                # ========== 能量分析：历史主方向 ==========
                total_energy = (S_hist ** 2).sum().item()
                top1_ratio = (S_hist[0] ** 2 / total_energy).item() if len(S_hist) > 0 else 0
                top3_ratio = (S_hist[:3] ** 2).sum().item() / total_energy if len(S_hist) >= 3 else 0
                top5_ratio = (S_hist[:5] ** 2).sum().item() / total_energy if len(S_hist) >= 5 else 0
                
                # 历史主方向与目标/锚点的对齐
                # delta_history: [out_dim, emb_size], t_vec_avg: [1, emb_size]
                # 先将目标向量映射到输出空间，再与历史主方向比较
                delta_history_t = delta_history @ t_vec_avg.T  # [out_dim, 1]
                delta_history_a = delta_history @ a_vec_avg.T  # [out_dim, 1]
                hist_t_align = (U_sel.T @ delta_history_t).norm().item() / (delta_history_t.norm().item() + 1e-8) if U_sel.shape[1] > 0 else 0
                hist_a_align = (U_sel.T @ delta_history_a).norm().item() / (delta_history_a.norm().item() + 1e-8) if U_sel.shape[1] > 0 else 0
                
                # 历史能量中被保护的比例
                protected_energy = ((U_sel @ U_sel.T @ delta_history).norm() / delta_history.norm()).item() if delta_history.norm() > 1e-8 else 0
                
                with open(config_path, "a") as f:
                    f.write(
                        f"    [HistEnergy] rank=1/1 | "
                        f"top1={top1_ratio:.4f} | top3={top3_ratio:.4f} | top5={top5_ratio:.4f} | "
                        f"align_t={hist_t_align:.4f} | align_a={hist_a_align:.4f} | "
                        f"protected={protected_energy:.4f}\n"
                    )
                # ==========================================
            else:
                P_hist = torch.eye(W.shape[0], device=device, dtype=W.dtype)
                with open(config_path, "a") as f:
                    f.write(f"    [HistEnergy] rank=0, no protected subspace\n")
        else:
            P_hist = torch.eye(W.shape[0], device=device, dtype=W.dtype)

        m_hist[name] = delta_coef * m_prev + (1 - delta_coef) * noise
        v_hist[name] = delta_coef * v_prev + (1 - delta_coef) * ((noise - m_hist[name]) ** 2)
    
        with open(config_path, "a") as f:
            f.write(f"\n")

        # ---- Retain ----
        layer_ret_embs = last_ret_embs
        sum_rr, n = [], 0
        for i in range(0, len(layer_ret_embs), chunk_size):
            chunk = layer_ret_embs[i:i + chunk_size]
            n += chunk.size(0)
            sum_rr.append((chunk.transpose(1, 2) @ chunk).sum(0))
        sum_rr = torch.stack(sum_rr).sum(0) / max(1, n)

        U, S, _ = torch.svd(sum_rr)
        mask = S < args.threshold
        if mask.sum() == 0:
            continue
        P = U[:, mask] @ U[:, mask].T
        
        retain_dims = mask.sum().item()
        total_dims = mask.numel()

        M = (sum_tt @ P + args.retain_scale * I).inverse()
        delta = (
            W @ (sum_at - sum_tt) @ P
            @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P)
            @ M
        )
        
        # ========== 应用 P_hist 前记录 ==========
        delta_before = delta.clone()
        
        delta = P_hist @ delta
        
        # ========== 编辑方向分析 ==========
        edit_stats = analyze_edit_direction(
            delta, delta_history, k_e, t_vec_avg, a_vec_avg, P, P_hist,
            name, step, config_path
        )
        
        # 额外：P_hist 造成的能量损失
        energy_loss = ((delta_before - delta).norm() / delta_before.norm()).item() if delta_before.norm() > 1e-8 else 0
        
        with open(config_path, "a") as f:
            f.write(
                f"    [Summary] retain_dims={retain_dims}/{total_dims} | "
                f"P_hist_loss={energy_loss:.4f} | "
                f"final_efficiency={edit_stats['edit_efficiency']:.4f}\n\n"
            )
        # ==================================

        delta_total = delta_history + delta
        global_hist_norm_sq += torch.norm(delta_total @ k_e) ** 2
        global_cur_norm_sq += torch.norm(delta @ k_e) ** 2
        edit_dict[name] = (W + delta).cpu()

    noise_e = abs(global_hist_norm_sq - global_cur_norm_sq)
    return edit_dict, m_hist, v_hist, noise_e


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--edit_ckpt", help="save history weight,m,v", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default="concept")
    parser.add_argument("--params", type=str, default="V")
    parser.add_argument("--aug_num", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--retain_scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--delta_coef", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=1.0)

    args = parser.parse_args()

    device = "cuda"
    seed_everything(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # ---- Load pipeline ----
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=dtype,)
    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.vae = None

    pipe.text_encoder.to(device)
    pipe.unet.to(device)

    basestate = {k: v.clone() for k, v in pipe.unet.state_dict().items()}
    m_hist = {}
    v_hist = {}

    # ---- SEQUENTIAL LOAD POINT ----
    if args.edit_ckpt and os.path.exists(args.edit_ckpt):
        if os.path.isfile(args.edit_ckpt):
            weight_path = args.edit_ckpt
            ckpt_dir = os.path.dirname(args.edit_ckpt)
            m_path = os.path.join(ckpt_dir, "m_hist.pt")
            v_path = os.path.join(ckpt_dir, "v_hist.pt")
            print(f"[INFO] Loading previous edit file: {args.edit_ckpt}")
        else:
            weight_path = os.path.join(args.edit_ckpt, "weight.pt")
            m_path = os.path.join(args.edit_ckpt, "m_hist.pt")
            v_path = os.path.join(args.edit_ckpt, "v_hist.pt")
            print(f"[INFO] Loading previous edit folder: {args.edit_ckpt}")

        if os.path.exists(weight_path):
            prev_weight = torch.load(weight_path, map_location="cpu")
            pipe.unet.load_state_dict(prev_weight, strict=False)
        else:
            print("[WARNING] weight.pt not found, using base model")

        if os.path.exists(m_path):
            m_hist = torch.load(m_path)
        else:
            print("[INFO] m_hist.pt not found, init empty dict")

        if os.path.exists(v_path):
            v_hist = torch.load(v_path)
        else:
            print("[INFO] v_hist.pt not found, init empty dict") 

    # ---- Parse inputs ----
    current_target = [x.strip() for x in args.target_concepts.split(",") if x.strip()]
    anchors = [x.strip() for x in args.anchor_concepts.split(",")]

    if len(anchors) == 1:
        anchors = anchors * len(current_target)

    retain_texts = [""]
    if args.retain_path:
        df = pd.read_csv(args.retain_path)
        df_retain = df[(df["id"] >= 201) & (df["id"] <= 600)]
        retain_texts = df_retain[args.heads].dropna().astype(str).tolist()

    torch.cuda.empty_cache()
    
    # ---- Edit ----
    edit_dict, m_hist, v_hist, noise_e = edit_model(
        args, pipe, basestate, current_target, anchors, retain_texts, m_hist, v_hist, device=device
    )

    os.makedirs(args.save_path, exist_ok=True)
    weight_path = os.path.join(args.save_path, "weight.pt")
    torch.save(edit_dict, weight_path)
    
    m_path = os.path.join(args.save_path, "m_hist.pt")
    v_path = os.path.join(args.save_path, "v_hist.pt")
    torch.save(m_hist, m_path)
    torch.save(v_hist, v_path)
    
    # ---- noise_E log path ----
    log_dir = os.path.dirname(os.path.dirname(args.save_path))
    match = re.search(r"step_(\d+)", args.save_path)
    step = int(match.group(1))
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, "noise_e_log.txt")
    with open(txt_path, "a") as f:
        f.write(f"step: {step}  noise_e: {noise_e:.8f}\n")

    values = []
    with open(txt_path, "r") as f:
        for line in f:
            if "noise_e:" in line:
                val = float(line.strip().split("noise_e:")[1])
                values.append(val)
    noise_E = sum(values) / len(values)
    noise_E_path = os.path.join(log_dir, "noise_E_log.txt")
    with open(noise_E_path, "a") as f:
        f.write(f"step: {step}  noise_E: {noise_E:.8f}\n")
    print(f"[INFO] noise_E updated: {noise_E:.8f}")
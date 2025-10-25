# trace_utils.py
import os, math, json, torch
from collections import defaultdict

def _basic_stats(t: torch.Tensor, topk=(0.0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)):
    t = t.detach().float().view(-1)
    if t.numel() == 0:
        return {}
    abs_t = t.abs()
    q = torch.quantile(t, torch.tensor([p/100 for p in topk], device=t.device))
    qa = torch.quantile(abs_t, torch.tensor([p/100 for p in topk], device=t.device))
    return {
        "numel": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "abs_p": {str(p): float(v) for p, v in zip(topk, qa.cpu())},
        "p": {str(p): float(v) for p, v in zip(topk, q.cpu())},
        "sparsity(==0)": float((t==0).float().mean().cpu()),
        "max_abs": float(abs_t.max().cpu()),
    }

def suggest_bits_from_range(max_abs, safety_ratio=0.999):
    """
    用「覆蓋比例」估位元數（對稱量化；不對齊 2 的冪也可先估）：
    找最小 b，使得 (2^(b-1)-1) >= max_abs / scale
    若你用 abs-max 量化，scale = (2^(b-1)-1) / max_abs
    這裡先回傳 b 的保守估計。
    """
    if max_abs <= 0:
        return 1
    # 以 abs-max 假設：讓 99.9% 的幅度落在可表示範圍
    # 其實你可用統計裡的 abs_p["99.9"] 代替 max_abs 再算更保守/務實
    for b in range(2, 17):  # 2~16 bits 搜尋
        q = (2**(b-1) - 1)
        if q >= max_abs:
            return b
    return 16

class ActivationTracer:
    """
    把各階段張量做統計（均值、標準差、分位數、稀疏度…）
    並可選擇把少量樣本 raw 值下採樣存檔。
    """
    def __init__(self, save_dir="runs/trace", sample_raw_every=0,
                 sample_merge=False,          # << 新增：是否合併存成一個檔
                 sample_cap_per_node=None):   # << 新增：每個節點最多保留多少元素（避免爆 RAM）
        self.buff = defaultdict(list)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.sample_raw_every = sample_raw_every
        self.sample_merge = sample_merge
        self.sample_cap_per_node = sample_cap_per_node
        self._step = 0

        # 合併模式下的暫存（每個節點一個 list）
        self._raw_cache = defaultdict(list) if self.sample_merge else None

    def log(self, name: str, t: torch.Tensor):
        stats = _basic_stats(t)
        if stats:
            self.buff[name].append(stats)

        # ↓↓↓ Raw sample 邏輯
        if self.sample_raw_every > 0 and (self._step % self.sample_raw_every == 0):
            if self.sample_merge:
                # 合併模式：把扁平向量暫存起來，dump() 再一次性存
                vec = t.detach().cpu().view(-1)
                if self.sample_cap_per_node is not None:
                    # 簡單裁切（你也可改成隨機取樣）
                    remain = self.sample_cap_per_node - sum(x.numel() for x in self._raw_cache[name])
                    if remain > 0:
                        self._raw_cache[name].append(vec[:remain])
                else:
                    self._raw_cache[name].append(vec)
            else:
                # 舊行為：每個 step 存一個檔
                raw_path = os.path.join(self.save_dir, f"{name.replace('/','_')}_step{self._step}.pt")
                torch.save(t.detach().cpu(), raw_path)

    def step(self):
        self._step += 1

    def dump(self, tag="summary"):
        # 1) 輸出統計 JSON（維持你原本的行為）
        out = {k: {
            "count": len(v),
            "mean_of_means": float(sum(d["mean"] for d in v)/len(v)),
            "mean_of_stds": float(sum(d["std"] for d in v)/len(v)),
            "global_max_abs": float(max(d["max_abs"] for d in v)),
            "suggest_bits_absmax": int(suggest_bits_from_range(max(d["max_abs"] for d in v))),
        } for k, v in self.buff.items()}
        path = os.path.join(self.save_dir, f"{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        # 2) 若開了合併模式，把每個節點的 raw 數值合併後存成一個 _all.pt
        if self.sample_merge and self._raw_cache:
            for name, chunks in self._raw_cache.items():
                if not chunks:
                    continue
                merged = torch.cat(chunks)  # 1D
                all_path = os.path.join(self.save_dir, f"{name.replace('/','_')}_all.pt")
                torch.save(merged, all_path)

        return path

# 在 main.py 或 trace_utils.py 後面
def install_encoder_hooks(model, tracer: "ActivationTracer"):
    """
    在每一層 EncoderLayer 裝 hook：
    - attn linears[0..2] 輸出（Q/K/V）
    - attn 輸出 context
    - FFN 中間與輸出
    """

    if getattr(model, "_tracer_installed", False):
        return
    
    # A) MultiHeadedAttention.forward 會回傳 (context)，並把 attn map 存在 self.attn
    #    我們無需改 forward，改在 EncoderLayer 的 sublayer 呼叫點附近裝 hook。
    for li, layer in enumerate(model.decoder.layers):  # Encoder 的每一層
        attn = layer.self_attn
        ffn  = layer.feed_forward

        # 1) Q/K/V projection 的輸出
        for idx, proj in enumerate(attn.linears[:3]):  # 0:Q,1:K,2:V
            proj.register_forward_hook(lambda m, inp, out, li=li, idx=idx:
                tracer.log(f"layer{li}/attn/{['Q','K','V'][idx]}", out))

        # 2) Attention 最後輸出（線性合併前 / 後 皆可）
        # attn.linears[-1] 是最後的線性投影
        attn.linears[-1].register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/attn/out", out))

        # 3) FFN：w1(x) 之後的 ReLU 輸出，以及 w2 之後
        ffn.w_1.register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/ffn/w1_out", out))
        ffn.w_2.register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/ffn/w2_out", out))

    setattr(model, "_tracer_installed", True)
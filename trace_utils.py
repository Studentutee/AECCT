# trace_utils.py
import os, math, json, torch
from collections import defaultdict

def _basic_stats(t: torch.Tensor, topk=(0.0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)):
    t = t.detach().float().reshape(-1)  # 支援非連續/expand 來源
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
                vec = t.detach().cpu().reshape(-1)  # 安全展平
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
def install_detailed_hooks(model, tracer: "ActivationTracer"):
    """
    新增「細分」trace（Softmax 前/後、head 合併前/後、LayerNorm in/out、ReLU in/out、以及 [|y|, s(y)] 輸入與 LPE 串接）。
    """
    import types, math, torch, torch.nn.functional as F
    if getattr(model, "_tracer_detailed_installed", False):
        return

    # ---------- 0) 記錄 [|y|, s(y)] 與 embedding / LPE 串接 ----------
    orig_forward = model.forward
    def fwd_with_input_trace(self, magnitude, syndrome):
        # [|y|, s(y)]（列Row對應節點維度；行Column對應 batch 維度）
        tracer.log("input/abs_y", magnitude)
        tracer.log("input/syndrome", syndrome)
        # [|y|, 1-2s(y)] 論文定義的模型實際輸入 h(y)
        h_y = torch.cat([magnitude, syndrome], dim=-1)
        tracer.log("input/h_y", h_y)
        # embedding 前乘法輸入（node_embed_before_cat）
        emb0 = torch.cat([magnitude, syndrome], dim=-1).unsqueeze(-1)
        node_embed = self.src_embed.unsqueeze(0) * emb0
        tracer.log("embed/node_embed", node_embed)

        # LPE 串接前/後
        lpe = self.lpe_proj(self.lpe)
        lpe = self.attn_lpe(lpe).unsqueeze(0)
        bached_lpe = lpe.expand(node_embed.size(0), lpe.size(1), lpe.size(2))
        tracer.log("embed/lpe_after_proj_attn", bached_lpe)
        emb_cat = torch.cat([node_embed, bached_lpe], dim=-1)
        tracer.log("embed/plus_SPE", emb_cat)

        # 照原本 forward 邏輯重做（避免重複計算，直接複用原 forward）
        return orig_forward(magnitude, syndrome)
    model.forward = types.MethodType(fwd_with_input_trace, model)

    # ---------- 1) Attention 細分（scores → softmax → context） ----------
    for li, layer in enumerate(model.decoder.layers):
        attn = layer.self_attn
        ffn  = layer.feed_forward

        # (a) Q/K/V projection 輸出（延續舊行為）
        for idx, proj in enumerate(attn.linears[:3]):
            proj.register_forward_hook(lambda m, inp, out, li=li, idx=idx:
                tracer.log(f"layer{li}/attn/{['Q','K','V'][idx]}", out))

        # (b) 取代 attention()，把 scores / softmax / context 全記下
        orig_attention = attn.attention
        def attention_hooked(self_, q, k, v, mask):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self_.d_k)
            tracer.log(f"layer{li}/attn/scores_in", scores)
            if mask is not None:
                scores = scores.masked_fill(mask.bool(), torch.finfo(scores.dtype).min)
            p_attn = F.softmax(scores, dim=-1)
            tracer.log(f"layer{li}/attn/softmax_out", p_attn)
            context = torch.matmul(p_attn, v)
            tracer.log(f"layer{li}/attn/context", context)
            if self_.dropout is not None:
                p_attn = self_.dropout(p_attn)
            return context, p_attn
        attn.attention = types.MethodType(attention_hooked, attn)

        # (c) 取代 hpsa()，記錄 head 合併前/後（pre/post concat）
        orig_hpsa = attn.hpsa
        def hpsa_hooked(self_, q, k, v, mask):
            out, attn_map = orig_hpsa(q, k, v, mask)
            # 這裡的 out = [B, H, N, Dh] 經轉置後才 concat；我們把「concat 前」先存（各 head 維度）
            tracer.log(f"layer{li}/attn/pre_concat", out)
            return out, attn_map
        attn.hpsa = types.MethodType(hpsa_hooked, attn)

        # (d) 最後線性投影前（post_concat）與投影後（原本就有 attn/out）
        attn.linears[-1].register_forward_pre_hook(lambda m, inp, li=li:
            tracer.log(f"layer{li}/attn/post_concat", inp[0]))
        attn.linears[-1].register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/attn/out", out))

        # ---------- 2) LayerNorm（兩個 sublayer 的 in/out + Encoder 結尾 LayerNorm） ----------
        layer.sublayer[0].norm.register_forward_pre_hook(lambda m, inp, li=li:
            tracer.log(f"layer{li}/norm/attn_in", inp[0]))
        layer.sublayer[0].norm.register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/norm/attn_out", out))

        layer.sublayer[1].norm.register_forward_pre_hook(lambda m, inp, li=li:
            tracer.log(f"layer{li}/norm/ffn_in", inp[0]))
        layer.sublayer[1].norm.register_forward_hook(lambda m, inp, out, li=li:
            tracer.log(f"layer{li}/norm/ffn_out", out))

        # ---------- 3) FFN 的 ReLU 前/後 ----------
        orig_ffn_forward = ffn.forward
        def ffn_forward_hooked(self_, x):
            w1 = self_.w_1(x)
            tracer.log(f"layer{li}/ffn/relu_in", w1)
            r  = F.relu(w1)
            tracer.log(f"layer{li}/ffn/relu_out", r)
            r  = self_.dropout(r)
            out = self_.w_2(r)
            tracer.log(f"layer{li}/ffn/w1_out", w1)
            tracer.log(f"layer{li}/ffn/w2_out", out)
            return out
        ffn.forward = types.MethodType(ffn_forward_hooked, ffn)

    # Encoder 結尾 LayerNorm（Post-LN）的 in/out
    enc = model.decoder
    if hasattr(enc, "norm"):
        enc.norm.register_forward_pre_hook(lambda m, inp:
            tracer.log("encoder/norm_end_in", inp[0]))
        enc.norm.register_forward_hook(lambda m, inp, out:
            tracer.log("encoder/norm_end_out", out))
    if hasattr(enc, "norm2"):
        # 中段的 LayerNorm（若存在）
        enc.norm2.register_forward_pre_hook(lambda m, inp:
            tracer.log("encoder/norm_mid_in", inp[0]))
        enc.norm2.register_forward_hook(lambda m, inp, out:
            tracer.log("encoder/norm_mid_out", out))

    setattr(model, "_tracer_detailed_installed", True)
# main.py  (with phase-select, resume, best-ckpt, periodic test, rich filenames)
import os, argparse, logging
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from configuration import Config, Code
from dataset import (
    ECC_Dataset, train, test, set_seed, Get_Generator_and_Parity, EbN0_to_std
)
from models import ECC_Transformer, freeze_weights


# ---------- helpers ----------
def build_sigma_list(k, n, ebn0_min=3.0, ebn0_max=7.0, steps=21):
    rate = float(k) / float(n)
    ebn0s = np.linspace(ebn0_min, ebn0_max, steps)
    return [EbN0_to_std(e, rate) for e in ebn0s]


def make_dataloaders(args: Config, runlen_train=128000, ebn0_test_list=(4, 5, 6)):
    # Train: zero codeword, AWGN with Eb/N0 in [3,7]
    sigma_train = build_sigma_list(args.code.k, args.code.n, 3.0, 7.0, 21)
    ds_train = ECC_Dataset(code=args.code, sigma=sigma_train, len=runlen_train, zero_cw=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)

    # Test: separate loaders per Eb/N0
    test_loaders = []
    for eb in ebn0_test_list:
        sigma = [EbN0_to_std(eb, float(args.code.k) / float(args.code.n))]
        ds_t = ECC_Dataset(code=args.code, sigma=sigma, len=args.test_batch_size * 20, zero_cw=True)
        test_loaders.append(DataLoader(ds_t, batch_size=args.test_batch_size,
                                       shuffle=False, num_workers=args.workers, pin_memory=True))
    return dl_train, test_loaders, ebn0_test_list


def save_ckpt(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    logging.info(f"Checkpoint saved: {path}")


def load_ckpt_if_any(model: torch.nn.Module, ckpt_path: str, strict=True):
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd, strict=strict)
        logging.info(f"Loaded checkpoint: {ckpt_path} (strict={strict})")
    elif ckpt_path:
        logging.warning(f"[resume] File not found: {ckpt_path} (skipped)")


def maybe_validate(model, device, test_loader_list, ebn0_list, epoch, val_every):
    if val_every > 0 and epoch % val_every == 0:
        logging.info(f"[Eval] epoch {epoch}: running test at Eb/N0 {ebn0_list}")
        test(model, device, test_loader_list, ebn0_list, min_FER=100)


def fmt_ckpt_tag(args: Config) -> str:
    """建立檔名用的模型/碼參數標籤"""
    ct = args.code.code_type if hasattr(args, "code") and hasattr(args.code, "code_type") else "CODE"
    n  = args.code.n if hasattr(args, "code") and hasattr(args.code, "n") else "N"
    k  = args.code.k if hasattr(args, "code") and hasattr(args.code, "k") else "K"
    return f"{ct}_n{n}_k{k}__Ndec{args.N_dec}_d{args.d_model}_h{args.h}"


def fmt_el(epoch: int, loss: float) -> str:
    """epoch+loss 標籤（for 檔名）"""
    return f"e{epoch}_loss{loss:.6f}"


# ---------- Stage 1 ----------
def stage1_train(args: Config, device, resume: str, epochs1: int, outdir: str, val_every: int):
    # Stage 1: standard transformer (no AAP)
    args.use_aap_linear_training = False
    args.use_aap_linear_inference = False

    model = ECC_Transformer(args).to(device)
    load_ckpt_if_any(model, resume, strict=True)

    train_loader, test_loader_list, ebn0_list = make_dataloaders(
        args, runlen_train=args.batch_size * 1000, ebn0_test_list=(4, 5, 6)
    )
    optim = Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(optim, T_max=epochs1, eta_min=args.eta_min)

    # Track/save best
    best_loss = float("inf")
    best_epoch = 0
    tag = fmt_ckpt_tag(args)
    best_any_path = os.path.join(outdir, "best_model")  # 共用名（覆蓋）
    best_stage1_clean = os.path.join(outdir, f"best_stage1_fp32__{tag}.pth")  # 乾淨名（覆蓋）
    final_clean = os.path.join(outdir, f"stage1_fp32__{tag}.pth")             # 最後一輪（覆蓋）
    last_loss = None

    for epoch in range(1, epochs1 + 1):
        LR = optim.param_groups[0]['lr']
        loss, _, _ = train(model, device, train_loader, optim, epoch, LR, args)
        sched.step()
        last_loss = float(loss)

        # save best on-the-fly (by training loss)
        if loss < best_loss:
            best_loss = float(loss)
            best_epoch = epoch
            logging.info(f"[P1] saving best_model with loss {best_loss:.6f} at epoch {best_epoch}")
            # 覆蓋：共用名 + 乾淨名
            torch.save(model.state_dict(), best_any_path)
            torch.save(model.state_dict(), best_stage1_clean)
            # 歷史歸檔：帶 epoch+loss
            best_stage1_arch = os.path.join(outdir, f"best_stage1_fp32__{tag}__{fmt_el(best_epoch, best_loss)}.pth")
            torch.save(model.state_dict(), best_stage1_arch)

        # periodic eval
        maybe_validate(model, device, test_loader_list, ebn0_list, epoch, val_every)

    # save last epoch model（覆蓋 + 歷史）
    save_ckpt(model.state_dict(), final_clean)
    final_arch = os.path.join(outdir, f"stage1_fp32__{tag}__{fmt_el(epochs1, last_loss if last_loss is not None else float('nan'))}.pth")
    save_ckpt(model.state_dict(), final_arch)

    # final eval
    test(model, device, test_loader_list, ebn0_list, min_FER=100)
    return final_clean


# ---------- Stage 2 ----------
def stage2_qat(args: Config, device, resume_from: str, resume_qat: str,
               epochs2: int, outdir: str, val_every: int):
    # Stage 2 (QAT): replace Linear with AAPLinearTraining
    args.use_aap_linear_training = True
    args.use_aap_linear_inference = False

    qat_model = ECC_Transformer(args).to(device)

    # init from Stage-1 FP32（strict=False 因為 AAP 多 delta 等參數）
    if resume_from:
        load_ckpt_if_any(qat_model, resume_from, strict=False)
    # resume from Stage-2 (QAT) ckpt
    if resume_qat:
        load_ckpt_if_any(qat_model, resume_qat, strict=True)

    train_loader, test_loader_list, ebn0_list = make_dataloaders(
        args, runlen_train=args.batch_size * 1000, ebn0_test_list=(4, 5, 6)
    )
    optim = Adam(qat_model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(optim, T_max=epochs2, eta_min=args.eta_min)

    # Track/save best
    best_loss = float("inf")
    best_epoch = 0
    tag = fmt_ckpt_tag(args)
    best_any_path = os.path.join(outdir, "best_model")                     # 共用名（覆蓋）
    best_qat_clean = os.path.join(outdir, f"best_stage2_qat__{tag}.pth")   # 乾淨名（覆蓋）
    qat_final_clean = os.path.join(outdir, f"stage2_qat__{tag}.pth")       # 最後一輪（覆蓋）
    last_loss = None

    for epoch in range(1, epochs2 + 1):
        LR = optim.param_groups[0]['lr']
        loss, _, _ = train(qat_model, device, train_loader, optim, epoch, LR, args)
        sched.step()
        last_loss = float(loss)

        if loss < best_loss:
            best_loss = float(loss)
            best_epoch = epoch
            logging.info(f"[P2] saving best_model (QAT) with loss {best_loss:.6f} at epoch {best_epoch}")
            # 覆蓋：共用名 + 乾淨名
            torch.save(qat_model.state_dict(), best_any_path)
            torch.save(qat_model.state_dict(), best_qat_clean)
            # 歷史歸檔
            best_qat_arch = os.path.join(outdir, f"best_stage2_qat__{tag}__{fmt_el(best_epoch, best_loss)}.pth")
            torch.save(qat_model.state_dict(), best_qat_arch)

        # periodic eval
        maybe_validate(qat_model, device, test_loader_list, ebn0_list, epoch, val_every)

    # save last epoch model（覆蓋 + 歷史）
    save_ckpt(qat_model.state_dict(), qat_final_clean)
    qat_final_arch = os.path.join(outdir, f"stage2_qat__{tag}__{fmt_el(epochs2, last_loss if last_loss is not None else float('nan'))}.pth")
    save_ckpt(qat_model.state_dict(), qat_final_arch)

    # Switch to inference (AAPLinearInference) and freeze ternary weights
    args.use_aap_linear_training = False
    args.use_aap_linear_inference = True
    infer_model = ECC_Transformer(args).to(device)
    load_ckpt_if_any(infer_model, qat_final_clean, strict=False)
    freeze_weights(infer_model, args)  # quantize to {-1,0,1}, set s_w
    infer_path = os.path.join(
        outdir,
        f"stage2_infer_frozen__{tag}__{fmt_el(best_epoch if best_epoch>0 else epochs2, best_loss if best_loss < float('inf') else (last_loss if last_loss is not None else float('nan')))}.pth"
    )
    save_ckpt(infer_model.state_dict(), infer_path)

    # final eval with frozen inference model
    test(infer_model, device, test_loader_list, ebn0_list, min_FER=100)
    return qat_final_clean, infer_path


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="AECCT Training (phase-select, resume, epochs)")
    # code settings
    parser.add_argument("--code_type", type=str, default="LDPC",
                        choices=["LDPC", "POLAR", "BCH", "CCSDS", "MACKAY"])
    parser.add_argument("--n", type=int, default=49)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--standardize", action="store_true",
                        help="use standard form for parity (if supported)")

    # model & train config
    parser.add_argument("--N_dec", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)

    # phase selection & epochs
    parser.add_argument("--enable_p1", action="store_true", help="run Phase 1 (FP32) training")
    parser.add_argument("--enable_p2", action="store_true", help="run Phase 2 (QAT) training")
    parser.add_argument("--epochs1", type=int, default=0, help="epochs for Phase 1 (override)")
    parser.add_argument("--epochs2", type=int, default=0, help="epochs for Phase 2 (override)")

    # periodic validation
    parser.add_argument("--val_every", type=int, default=200, help="run test() every N epochs (0=disable)")

    # resume controls
    parser.add_argument("--resume1", type=str, default="", help="checkpoint path to resume Phase 1")
    parser.add_argument("--resume2", type=str, default="", help="checkpoint path to resume Phase 2 (QAT)")
    parser.add_argument("--from_stage1", type=str, default="", help="use this Stage-1 ckpt to init Stage-2")

    # output root
    parser.add_argument("--outdir", type=str, default="runs", help="output dir for checkpoints & logs")

    args_cli = parser.parse_args()

    # default: run both if neither specified
    if not (args_cli.enable_p1 or args_cli.enable_p2):
        args_cli.enable_p1 = True
        args_cli.enable_p2 = True

    # logging
    os.makedirs(args_cli.outdir, exist_ok=True)
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args_cli.outdir, f"train_{timestr}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    # Build Config & Code
    args = Config()
    args.N_dec = args_cli.N_dec
    args.d_model = args_cli.d_model
    args.h = args_cli.h
    args.batch_size = args_cli.batch_size
    args.test_batch_size = args_cli.test_batch_size
    args.workers = args_cli.workers
    args.seed = args_cli.seed
    args.lr = args_cli.lr
    args.eta_min = args_cli.eta_min
    args.standardize = args_cli.standardize

    # save path
    args.path = os.path.join(args_cli.outdir, timestr)
    os.makedirs(args.path, exist_ok=True)

    # build code matrices (loads from dataset.CODES_PATH, typically ./codes)
    code = Code(n=args_cli.n, k=args_cli.k, code_type=args_cli.code_type)
    G, H = Get_Generator_and_Parity(code, standard_form=args_cli.standardize)
    # 與原版一致：存成 long（相容索引/遮罩），G 轉置成列向量在前
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code

    # seed & device
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    # epochs
    epochs1 = args_cli.epochs1 if args_cli.epochs1 > 0 else 1000
    epochs2 = args_cli.epochs2 if args_cli.epochs2 > 0 else 1000

    # run
    p1_ckpt = ""
    if args_cli.enable_p1:
        p1_ckpt = stage1_train(
            args, device,
            resume=args_cli.resume1,
            epochs1=epochs1,
            outdir=args.path,
            val_every=args_cli.val_every
        )

    if args_cli.enable_p2:
        init_for_p2 = args_cli.from_stage1 or p1_ckpt
        qat_ckpt, frozen_ckpt = stage2_qat(
            args, device,
            resume_from=init_for_p2,
            resume_qat=args_cli.resume2,
            epochs2=epochs2,
            outdir=args.path,
            val_every=args_cli.val_every
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()

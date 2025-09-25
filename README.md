# Accelerating Error Correction Code Transformers
Compact training pipeline with two-phase training (FP32 → QAT), manual checkpoint control, per-phase epochs, periodic validation, and rich checkpoint filenames (code + model shape + epoch + best loss).

## Install

`pip install -r requirements.txt`

## Quick Start (LDPC 49/24, 6 layers, d_model=128)
Train both phases in one go:

`python main.py --enable_p1 --enable_p2 --code_type LDPC --n 49 --k 24 --N_dec 6 --d_model 128`

Only Phase 1 (FP32), 500 epochs:

`python main.py --enable_p1 --epochs1 500 --code_type LDPC --n 49 --k 24 --N_dec 6 --d_model 128`

## Two-Phase Training (FP32 → QAT)
# Phase 1 (FP32)

`python main.py --enable_p1 --epochs1 800 --code_type LDPC --n 49 --k 24 --N_dec 6 --d_model 128`

# Phase 2 (QAT)
Initialize from Phase-1 checkpoint:

`python main.py --enable_p2 --from_stage1 runs/<ts>/best_stage1_fp32__LDPC_n49_k24__Ndec6_d128_h8.pth --epochs2 300`

(Use `--from_stage1` for Phase-1 weights. Use `--resume2` only for resuming Phase-2.)

## Resume / Continue
Resume Phase 1:

`python main.py --enable_p1 --resume1 runs/<ts>/best_stage1_fp32__LDPC_n49_k24__Ndec6_d128_h8.pth --epochs1 200 --code_type LDPC --n 49 --k 24 --N_dec 6 --d_model 128`

Resume Phase 2 (QAT):

`python main.py --enable_p2 --resume2 runs/<ts>/stage2_qat__LDPC_n49_k24__Ndec6_d128_h8.pth --epochs2 300 --code_type LDPC --n 49 --k 24 --N_dec 6 --d_model 128`

⚠️ The model shape and code parameters (N_dec, d_model, h, n, k, code_type) must match the checkpoint you load (strict load).

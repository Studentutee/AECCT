
import numpy as np
import torch
import os
import torch
import random
from torch.utils import data
import logging
import time
from tqdm import tqdm

from configuration import Config

# FP8 native helpers
try:
    from te_fp8_utils import is_te_available, build_fp8_recipe, convert_linear_to_te, fp8_context
except Exception as _e:
    is_te_available = None
    build_fp8_recipe = None
    convert_linear_to_te = None
    fp8_context = None

CODES_PATH = "codes/"

def Read_pc_matrixrix_alist(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(
            lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip('\n'), dtype=int, sep=' ')
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H
#############################################
def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
        idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]

def get_standard_form(pc_matrix_):
    pc_matrix = pc_matrix_.copy().astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist()+[min(pc_matrix.shape) - 1,next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist()+[next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)
#############################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_Generator_and_Parity(code, standard_form = False):
    n, k = code.n, code.k
    path_pc_mat = os.path.join(CODES_PATH, f'{code.code_type}_N{str(n)}_K{str(k)}')
    if code.code_type in ['POLAR', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat+'.txt')
    elif code.code_type in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = Read_pc_matrixrix_alist(path_pc_mat+'.alist')
    else:
        raise Exception(f'Wrong code {code.code_type}')
    if standard_form and code.code_type not in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = get_standard_form(ParityMatrix).astype(int)
        GeneratorMatrix = np.concatenate([np.mod(-ParityMatrix[:, min(ParityMatrix.shape):].transpose(),2),np.eye(k)],1).astype(int)
    else:
        GeneratorMatrix = get_generator(ParityMatrix)
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0) and np.sum(GeneratorMatrix) > 0
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)



##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##################################################################


class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR, config: Config):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = cum_loss = 0.
    t = time.time()
    batch_idx = 0
    for m, x, z, y, magnitude, syndrome in tqdm(train_loader, position=0, leave=True, desc="Training"):
        z_mul = (y * bin_to_sign(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
        batch_idx += 1
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100, tracer=None,
         precision: str = "fp32", measure_tp: bool = False, warmup: int = 10,
         tp_include_loss: bool = False, fp8_native: bool = False,
         fp32_strict: bool = False):
    """
    precision: fp32 | fp16 | bf16 | int8 | e5m2 | e4m3
    measure_tp: only measure GPU forward (+hard decision) using CUDA events.
    warmup: number of batches ignored for throughput stats.
    tp_include_loss: include loss in timing window (default False).
    fp8_native: if True and precision in {e5m2,e4m3}, require TransformerEngine and use native FP8.
    fp32_strict: disable TF32 (use true FP32 math)
    """
    model.eval()

    # ---- FP32 vs TF32 control (Ampere/Hopper 有效；Turing 前無效) ----
    if precision == "fp32":
        if fp32_strict:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")  # highest = 禁用 TF32
            logging.info("[Precision] FP32(strict) enabled: TF32 disabled.")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")     # high = 允許 TF32
            logging.info("[Precision] FP32(allow TF32) enabled.")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            logging.info(f"[Device] capability={cap} (>=8.x has TF32).")

    # ---- Precision contexts ----
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else None)
    use_fp8 = precision in ("e5m2", "e4m3")
    fp8_recipe = None

    # Native FP8 setup (one-time conversion)
    if use_fp8:
        if not fp8_native:
            raise RuntimeError("Requested FP8 precision but --fp8_native is False. Set --fp8_native to enable native FP8 (TransformerEngine).")
        if is_te_available is None:
            raise RuntimeError("TransformerEngine helpers not importable. Ensure te_fp8_utils.py is present.")
        ok, err = is_te_available()
        if not ok:
            raise RuntimeError(f"TransformerEngine not available: {err}")
        # convert Linear -> TE Linear only if not already converted
        if not getattr(model, "_te_converted", False):
            convert_linear_to_te(model)
            setattr(model, "_te_converted", True)
        fp8_recipe = build_fp8_recipe(precision)

    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()

    # Throughput accumulators
    total_ms = 0.0
    total_samples_for_tp = 0
    seen_batches = 0
    start_ev = torch.cuda.Event(enable_timing=True) if (measure_tp and torch.cuda.is_available()) else None
    end_ev   = torch.cuda.Event(enable_timing=True) if (measure_tp and torch.cuda.is_available()) else None

    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.0

            pbar_total = int(min_FER) if (min_FER and min_FER > 0) else None
            pbar = tqdm(total=pbar_total, desc=f"Eval Eb/N0={EbNo_range_test[ii]} dB",
                        unit="err", leave=False)

            stop = False
            while not stop:
                for m, x, z, y, magnitude, syndrome in test_loader:
                    magnitude = magnitude.to(device, non_blocking=True)
                    syndrome  = syndrome.to(device, non_blocking=True)
                    y         = y.to(device, non_blocking=True)
                    x_dev     = x.to(device, non_blocking=True)

                    if tracer is not None:
                        tracer.log("input/abs_y", magnitude)
                        tracer.log("input/syndrome", syndrome)
                        tracer.log("input/y", y)

                        emb0 = torch.cat([magnitude, syndrome], dim=-1).unsqueeze(-1)
                        node_embed = model.src_embed.unsqueeze(0) * emb0
                        tracer.log("embed/node_embed", node_embed)

                        lpe = model.lpe_proj(model.lpe)
                        lpe = model.attn_lpe(lpe).unsqueeze(0)
                        bached_lpe = lpe.expand(node_embed.size(0), lpe.size(1), lpe.size(2))
                        embed_plus_spe = torch.cat([node_embed, bached_lpe], dim=-1)
                        tracer.log("embed/plus_SPE", embed_plus_spe)

                    # ---- Timing window: ONLY forward (+ optional loss) ----
                    if start_ev is not None:
                        torch.cuda.synchronize()
                        start_ev.record()

                    if use_fp8:
                        with fp8_context(fp8_recipe):
                            z_pred = model(magnitude, syndrome)
                    elif use_amp:
                        with torch.amp.autocast('cuda', dtype=amp_dtype):
                            z_pred = model(magnitude, syndrome)
                    else:
                        z_pred = model(magnitude, syndrome)

                    # Produce x_pred for metrics (hard decision). Loss optional.
                    z_mul = (y * bin_to_sign(x_dev))
                    if tp_include_loss:
                        loss, x_pred = model.loss(-z_pred, z_mul, y)
                    else:
                        # mimic model.loss's x_pred branch without computing BCE
                        x_pred = ( -z_pred * torch.sign(y) > 0 ).float()

                    if end_ev is not None:
                        end_ev.record()
                        torch.cuda.synchronize()
                        if seen_batches >= warmup:
                            total_ms += start_ev.elapsed_time(end_ev)
                            total_samples_for_tp += x.shape[0]
                        seen_batches += 1

                    if tracer is not None:
                        tracer.step()

                    # ---- Metrics accumulation (outside timing) ----
                    bs = x.shape[0]
                    if tp_include_loss:
                        test_loss += loss.item() * bs
                    ber_batch = BER(x_pred, x_dev)
                    fer_batch = FER(x_pred, x_dev)
                    test_ber += ber_batch * bs
                    test_fer += fer_batch * bs
                    cum_count += bs

                    pbar.update(int(round(fer_batch * bs)))

                    if ((min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or
                        cum_count >= 1e9):
                        if cum_count >= 1e9:
                            logging.info(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                        else:
                            logging.info(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                        stop = True
                        break
            pbar.close()

            cum_samples_all.append(cum_count)
            test_loss_list.append((test_loss / cum_count) if tp_include_loss else 0.0)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            logging.info(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')

        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno) in zip(test_loss_list, EbNo_range_test)]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno) in zip(test_loss_fer_list, EbNo_range_test)]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, elem) for (elem, ebno) in zip(test_loss_ber_list, EbNo_range_test)]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.4e}'.format(ebno, -np.log(elem)) for (elem, ebno) in zip(test_loss_ber_list, EbNo_range_test)]))

    if measure_tp and total_ms > 0:
        secs = total_ms / 1000.0
        sps  = total_samples_for_tp / secs
        logging.info(f"[Throughput] GPU forward-only: {sps:.2f} samples/s "
                     f"(ignored first {warmup} batches; "
                     f"{'incl. loss' if tp_include_loss else 'excl. loss'})")

    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list

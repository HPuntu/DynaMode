import os
import time
from huggingface_hub import hf_hub_download
import mdtraj
import tempfile
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import glob
import mdtraj as md
import yaml
import torch


def validate_batch(batch, required):
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing batch keys: {missing}")


def as_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"batch['{name}'] must be a torch.Tensor, got {type(value).__name__}")
    return value


def check_1d(t, name, B=None):
    if t.ndim != 1:
        raise ValueError(f"batch['{name}'] must be 1D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] must have shape ({B},), got {tuple(t.shape)}")


def check_2d(t, name, B=None, L=None):
    if t.ndim != 2:
        raise ValueError(f"batch['{name}'] must be 2D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] first dim must be {B}, got {t.shape[0]}")
    if L is not None and t.shape[1] != L:
        raise ValueError(f"batch['{name}'] second dim must be {L}, got {t.shape[1]}")


def check_3d(t, name, B=None, L=None, C=None):
    if t.ndim != 3:
        raise ValueError(f"batch['{name}'] must be 3D, got shape {tuple(t.shape)}")
    if B is not None and t.shape[0] != B:
        raise ValueError(f"batch['{name}'] first dim must be {B}, got {t.shape[0]}")
    if L is not None and t.shape[1] != L:
        raise ValueError(f"batch['{name}'] second dim must be {L}, got {t.shape[1]}")
    if C is not None and t.shape[2] != C:
        raise ValueError(f"batch['{name}'] last dim must be {C}, got {t.shape[2]}")


def check_res_type(t, name, B, L):
    if t.ndim == 2:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(f"batch['{name}'] must have shape ({B}, {L}), got {tuple(t.shape)}")
        return
    if t.ndim == 3:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(
                f"batch['{name}'] first two dims must be ({B}, {L}), got {tuple(t.shape)}"
            )
        return
    raise ValueError(
        f"batch['{name}'] must be residue indices (B, L) or one-hot (B, L, C), got {tuple(t.shape)}"
    )


def check_residue_feature(t, name, B, L):
    if t.ndim == 2:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(f"batch['{name}'] must have shape ({B}, {L}), got {tuple(t.shape)}")
        return
    if t.ndim == 3:
        if t.shape[0] != B or t.shape[1] != L:
            raise ValueError(
                f"batch['{name}'] first two dims must be ({B}, {L}), got {tuple(t.shape)}"
            )
        return
    raise ValueError(f"batch['{name}'] must be 2D or 3D, got {tuple(t.shape)}")


def retrieve_mdcath_h5(id, path, retries=5, base_delay=1, local_files_only=False):
    fname = f"mdcath_dataset_{id}.h5"
    if path:
        # Check root of cache directory
        p1 = os.path.join(path, fname)
        if os.path.exists(p1): 
            return p1
        
        # Check 'data' subfolder (common HF structure)
        p2 = os.path.join(path, 'data', fname)
        if os.path.exists(p2): 
            return p2
    
    if local_files_only:
        raise FileNotFoundError(f"Offline mode: File {fname} not found in {path} or {path}/data")

    for attempt in range(1, retries+1):
        try:
            return hf_hub_download(
                repo_id="compsciencelab/mdCATH", 
                filename=fname,
                subfolder='data',
                local_dir=path,
                repo_type="dataset",
                local_files_only=local_files_only
            )
        except Exception as e:
            if attempt == retries:
                raise
            delay = base_delay * (2 ** (attempt-1))
            print(f"hf_hub_download failed (attempt {attempt}/{retries}): {e}. retrying in {delay}s...")
            time.sleep(delay)

def write_pdb_frame_mdtraj(pdb_str, coords, frame_idx, out_pdb):
    # MDTraj needs a topology file, so write the PDB string to disk
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb_str)
        topo_path = f.name

    if frame_idx == None:
        print("saving topology native pdb")
        topo = mdtraj.load_pdb(topo_path)
        topo.save_pdb(out_pdb)
    else:
        # MDTraj uses nanometers, not Å
        traj = mdtraj.Trajectory(
            xyz=coords[frame_idx:frame_idx+1] / 10.0,
            topology=mdtraj.load_pdb(topo_path).topology
        )
        traj.save_pdb(out_pdb)

def pad_and_interleave(matrices, block_size=5, pad_value=-1, pad_rows=True, pad_cols=True):
    '''Concise: pad to max shape (rows/cols optional) and interleave blocks.'''
    mats = [np.asarray(m) for m in matrices]
    if not mats:
        return np.empty((0,0))

    # choose dtype that fits all arrays and pad_value
    dtype = np.result_type(*(m.dtype for m in mats), np.array(pad_value).dtype)
    mats = [m.astype(dtype, copy=False) for m in mats]

    max_rows = max(m.shape[0] for m in mats)
    max_cols = max(m.shape[1] for m in mats)

    target_rows = max_rows if pad_rows else None
    target_cols = max_cols if pad_cols else None

    def pad(m):
        rpad = (target_rows - m.shape[0]) if (target_rows is not None) else 0
        cpad = (target_cols - m.shape[1]) if (target_cols is not None) else 0
        return np.pad(m, ((0, max(0, rpad)), (0, max(0, cpad))),
                      mode='constant', constant_values=pad_value) if (rpad>0 or cpad>0) else m

    mats = [pad(m) for m in mats]

    ncols = mats[0].shape[1]
    pieces = []
    for start in range(0, (max_rows if pad_rows else max(m.shape[0] for m in mats)), block_size):
        end = start + block_size
        for m in mats:
            chunk = m[start:end]
            if chunk.size:
                if chunk.shape[1] != ncols:
                    raise ValueError("column count mismatch after padding")
                pieces.append(chunk)

    return np.vstack(pieces) if pieces else np.empty((0, ncols))

def maintain_cache_size(cache_dir, max_size_gb=50):
    '''
    Deletes the oldest files in cache_dir until total size is under max_size_gb.
    Safe to run during training (Linux handles open file deletion gracefully).
    '''
    # 1. Get all h5 files
    files = glob.glob(os.path.join(cache_dir, "*.h5"))
    if not files:
        files = glob.glob(os.path.join(cache_dir + "/data", "*.h5"))
    if not files:
        print(f"ERROR: couldnt find h5 files in cache path to clear: {cache_dir} + /data")
        return

    # 2. Calculate current size
    total_size = sum(os.path.getsize(f) for f in files)
    max_size_bytes = max_size_gb * (1024**3)

    if total_size < max_size_bytes:
        return # Safe, do nothing

    # 3. Sort by Access Time (Oldest first)
    # Using getmtime (modification) is often safer on clusters than getatime (access)
    files.sort(key=os.path.getmtime) 

    print(f"Cleaning Cache: Current {total_size / (1024**3):.2f} GB > Limit {max_size_gb} GB")

    # 4. Delete oldest until we are under the limit
    for f in files:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            total_size -= size
            #print(f"Deleted {os.path.basename(f)}") # Optional logging
            
            if total_size < max_size_bytes:
                break
        except OSError:
            pass # File might be locked or already gone, skip it

def no_spill_stratified_split(dataset, test_size=0.1, val_size=0.1, seed=42):
    '''
    Splits mdCATH dataset preventing domain leakage.
    Group = Domain ID (index_map[0]) -> Keeps all temps/reps of a domain together.
    Stratify = Temperature (index_map[1]) -> Ensures balanced temps across sets.
    '''

    # We use the Domain ID as the Group (to prevent leakage)
    groups = np.array([item[0] for item in dataset.index_map])
    
    # We use Temperature as the Stratification label (to ensure balance)
    y = np.array([item[1] for item in dataset.index_map])
    
    # 2. First Split: Separate Test Set
    # StratifiedGroupKFold requires n_splits, so we convert size to splits
    n_splits_test = int(round(1 / test_size))
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    
    # Generate the splits (we only need the first fold)
    # We pass np.zeros for X because we only care about y and groups
    split_gen = sgkf.split(np.zeros(len(y)), y, groups)
    train_val_idx, test_idx = next(split_gen)

    # 3. Second Split: Separate Validation from the remaining Train/Val
    # We must subset the metadata to the remaining indices
    y_remaining = y[train_val_idx]
    groups_remaining = groups[train_val_idx]
    
    # Adjust val_size to be relative to the remaining data
    # e.g. if Test=0.1 (10%), Val=0.1 (10%), then Val is ~11% of the remaining 90%
    val_rel = val_size / (1.0 - test_size)
    n_splits_val = int(round(1.0 / val_rel))
    
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    
    split_gen_val = sgkf_val.split(np.zeros(len(y_remaining)), y_remaining, groups_remaining)
    train_idx_rel, val_idx_rel = next(split_gen_val)
    
    # Map relative indices back to original indices
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return train_idx, val_idx, test_idx

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_phi_psi_from_files(traj):

    _, phi_gt = md.compute_phi(traj)  # phi_angles.shape == (n_frames, n_phi)
    _, psi_gt = md.compute_psi(traj)  # psi_angles.shape == (n_frames, n_psi)
    print(phi_gt.shape)
    n_frames = phi_gt.shape[0]
    n_res = traj.n_residues  # 327

    # create per-residue matrices filled with NaN
    phi_per_res = np.full((n_frames, n_res), np.nan)
    psi_per_res = np.full((n_frames, n_res), np.nan)

    # phi_angles columns correspond to residues 1..n_res-1  -> place into phi_per_res[:, 1:]
    phi_per_res[:, 1:] = np.rad2deg(phi_gt)    # shape aligns: (n_frames, n_res-1)

    # psi_angles columns correspond to residues 0..n_res-2 -> place into psi_per_res[:, :-1]
    psi_per_res[:, :-1] = np.rad2deg(psi_gt)

    # Now same-residue valid mask (both not nan)
    valid_mask = ~np.isnan(phi_per_res) & ~np.isnan(psi_per_res)  # shape (n_frames, n_res)

    # collect paired values (flatten across frames if plotting entire set)
    phi_same = phi_per_res[valid_mask]
    psi_same = psi_per_res[valid_mask]

    return phi_same, psi_same

def build_spectral_mask(mask, torsion_mask, top_k, channels, is_dct):
    '''Build per-element mask for the flattened spectral volume.

    Args:
        mask: (B, L) residue validity mask.
        torsion_mask: (B, L, n_angle_channels) or None for coords-only.
        top_k: Number of frequency bins.
        channels: Total feature channels (e.g. 7 for coords+angles, 3 for coords-only).
        is_dct: If True, real-valued DCT; if False, complex DFT (×2).
    '''
    mask_coords = mask.unsqueeze(-1).expand(-1, -1, 3)
    if torsion_mask is not None:
        feature_mask = torch.cat([mask_coords, torsion_mask], dim=-1)
    else:
        feature_mask = mask_coords

    if is_dct:
        full = feature_mask.unsqueeze(2).expand(-1, -1, top_k, -1)
        return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)
    else:
        full = feature_mask.unsqueeze(2).unsqueeze(-1).expand(-1, -1, top_k, -1, 2)
        return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)


def flatten_yaml_config(path):
    raw = yaml.safe_load(open(path)) or {}
    config = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            config.update(value)
        else:
            config[key] = value
    return config


def read_config(config_path):

    config = {}
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)
        config.update(flatten_yaml_config(config_path))

    # Type coercions for YAML
    for float_key in ["max_lr", "min_snr_gamma", "smoothing_sigma", "shift_value", "guidance_scale", "shelf_value", "coord_scale", "bending_lambda", "geometry_lambda", "aniso_gamma", "rmsf_lambda", "dc_lambda"]:
        if float_key in config and config[float_key] is not None:
            if float_key == "shift_value" and isinstance(config[float_key], str) and config[float_key].strip().lower() == "auto":
                continue
            config[float_key] = float(config[float_key])
    for int_key in ["epochs", "batch_size", "num_workers", "top_k_freqs", "hidden_dim", "freq_hidden_size",
                        "spectral_modes",
                        "seq_embed_dim",
                        "ss_embed_dim",
                        "num_layers", "num_heads", "num_steps",
                        "num_ode_steps", "max_val_batches",
                        "geometry_warmup_start", "geometry_warmup_epochs",
                        "geometry_decay_start", "geometry_decay_epochs",
                        "rmsf_warmup_start", "rmsf_warmup_epochs",
                        "dc_start_epoch",
                        "prefetch_factor", "dataloader_timeout",
                        "diagnostic_low_k_modes", "band_low_modes", "band_mid_modes",
                        "rmsf_position_bins", "diversity_samples", "diversity_batches", "diversity_seed"]:
        if int_key in config:
            config[int_key] = int(config[int_key])

    if "low_k_correction_modes" in config and config["low_k_correction_modes"] is not None:
        value = config["low_k_correction_modes"]
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)
        config["low_k_correction_modes"] = value


    return config

def ca_bond_lengths(coords, eps=1e-8):
    '''Computes adjacent Ca-Ca bond lengths'''
    return torch.linalg.vector_norm(coords[..., 1:, :] - coords[..., :-1, :], dim=-1).clamp_min(eps)

def ca_bond_dirs(coords, eps=1e-8):
    '''Computes directions of adjacent Ca-Ca bonds'''
    bond = coords[..., 1:, :] - coords[..., :-1, :]
    return bond / torch.linalg.vector_norm(bond, dim=-1, keepdim=True).clamp_min(eps)

def chain_from_anchor_dirs_lengths(anchor, dirs, lengths):
    '''Reconstructs a chain of coordinates from an anchor point, bond directions, and bond lengths.'''
    steps = dirs * lengths.unsqueeze(-1)
    tail = anchor.unsqueeze(2) + torch.cumsum(steps, dim=2)
    return torch.cat([anchor.unsqueeze(2), tail], dim=2)

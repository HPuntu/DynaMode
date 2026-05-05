import os
import hashlib
import tempfile
import zarr
import torch
import numpy as np
import random
import mdtraj as md
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn.functional as F

from disco.dataloader.features import dssp_to_onehot, DSSP_STATES, compute_native_dssp_onehot



def stable_seed(*args):
    s = "_".join(map(str, args))
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2**32


def _open_zarr_root_or_store(zarr_path, label):
    try:
        root = zarr.open_consolidated(zarr_path, mode='r')
        return root, root.store
    except KeyError:
        print(f"WARNING: No .zmetadata found in {zarr_path}. Falling back to un-consolidated open.")

    try:
        root = zarr.open(zarr_path, mode='r')
        return root, root.store
    except zarr.errors.PathNotFoundError:
        if not os.path.isdir(zarr_path):
            raise

        store = zarr.DirectoryStore(zarr_path)
        try:
            has_keys = any(True for _ in store.keys())
        except Exception:
            has_keys = False
        if not has_keys:
            raise

        print(
            f"WARNING: {label} store at {zarr_path} has no root .zgroup metadata. "
            "Falling back to store-level group access."
        )
        return None, store


def _open_group_from_root_or_store(root, store, key):
    if root is not None:
        return root[key]
    return zarr.open(store=store, path=key, mode='r')


class FeaturizerWindowZarr:
    '''
    Featurizer for Spectral Volume Diffusion using pre-computed features in zarr files
    '''
    def __init__(self, window_size=256, include_angles=True, coords_type="ca"):
        self.window_size = window_size
        self.include_angles = include_angles
        self.coords_type = coords_type.lower()

    @staticmethod
    def _pad_tensor(t, pad_len, spatial_dim):
        if pad_len == 0:
            return t
        pads = [0] * (2 * t.ndim)
        dist_from_last = t.ndim - 1 - spatial_dim
        right_pad_idx = dist_from_last * 2 + 1
        pads[right_pad_idx] = pad_len
        return F.pad(t, tuple(pads))

    def get_time_window(self, total_frames, rng=None):
        if total_frames <= self.window_size: 
            return 0, int(total_frames)
        start = rng.integers(0, total_frames - self.window_size)
        return start, start + self.window_size

    def _get_ca_static_edges(self, n_res):
        # Operates on residue indices, so it works perfectly for both CA and BB
        src, dst = [], []
        for hop in [1, 2, 3]:
            for i in range(n_res - hop):
                src.extend([i, i + hop])
                dst.extend([i + hop, i])
                
        if not src: return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def process(self, coords, native_coords, res_type, dssp=None, angles=None, native_angles=None, torsion_mask=None):
        T_frames = coords.shape[0]
        n_res = coords.shape[1]

        # FLATTEN COORDINATE DIMENSIONS -> (T, L, C) and (L, C)
        # CA: (T, L, 3)   -> unchanged   C=3
        # BB: (T, L, 4, 3) -> (T, L, 12) C=12
        # =====================================================================
        if self.coords_type == 'bb' and coords.ndim == 4:
            T_f, L_f, A, xyz = coords.shape
            coords = coords.reshape(T_f, L_f, A * xyz)          # (T, L, 12)
            native_coords = native_coords.reshape(L_f, A * xyz)  # (L, 12)
        # CA coords are already (T, L, 3) - no reshape needed
        # =====================================================================

        # 1. Edges
        static_edges = self._get_ca_static_edges(n_res)

        # 2. Time Padding logic
        diff = self.window_size - T_frames if T_frames < self.window_size else 0
        time_mask = torch.ones(self.window_size)
        
        if diff > 0:
            time_mask = torch.cat([torch.ones(T_frames), torch.zeros(diff)])
            # This repeat logic naturally handles both [1, L, 3] and [1, L, 4, 3]
            coords_pad = coords[-1].unsqueeze(0).repeat(diff, *([1] * (coords.dim() - 1)))
            coords = torch.cat([coords, coords_pad], dim=0)

        # 3. Build Base Dictionary
        out = {
            "coords": coords,               
            "native_coords": native_coords, 
            "res_type": res_type,
            "dssp": dssp,
            "time_mask": time_mask,         
            "static_edges": static_edges    
        }

        # 4. Optionally Process and Pad Angles
        if self.include_angles and angles is not None:
            if diff > 0:
                angles_pad = angles[-1].unsqueeze(0).repeat(diff, 1, 1)
                angles = torch.cat([angles, angles_pad], dim=0)

            out["angles"] = angles               
            out["native_angles"] = native_angles 
            out["torsion_mask"] = torsion_mask   

        return out

    def collate_fn(self, batch):
        flat_batch = [item for sublist in batch if sublist for item in sublist if item]
        if not flat_batch: raise RuntimeError("All samples in batch were invalid")

        max_L = max(b["res_type"].shape[0] for b in flat_batch)
        collated = {}

        # spatial_dim maps to the L dimension
        sp_dim_map = {
            "coords": 1, "native_coords": 0, "res_type": 0,
            "dssp": 0,
            "residue_idx": 0,
            "rmsf_prior": 0,   # (L,) per-residue NMA prior, padded along L
            "dc_baseline_per_res": 0,  # (L, C) per-residue DC baseline in spectrum units
        }
        if self.include_angles:
            sp_dim_map.update({"angles": 1, "native_angles": 0, "torsion_mask": 0})

        for key in flat_batch[0].keys():
            if key in ["temp", "time_mask", "t", "dataset_idx", "win_pos"]:
                collated[key] = torch.stack([torch.as_tensor(b[key]) for b in flat_batch])
            elif key == "static_edges":
                continue
            elif key in sp_dim_map:
                if not all(key in b for b in flat_batch):
                    continue
                padded = []
                for b in flat_batch:
                    L_current = b["res_type"].shape[0]
                    pad_len = max_L - L_current
                    # Assuming _pad_tensor correctly pads along the provided spatial_dim
                    padded.append(self._pad_tensor(b[key], pad_len, spatial_dim=sp_dim_map[key]))
                collated[key] = torch.stack(padded)
        
        batched_edges = []
        for batch_idx, b in enumerate(flat_batch):
            offset = batch_idx * max_L
            offset_edges = b["static_edges"] + offset 
            batched_edges.append(offset_edges)
            
        collated["static_edges"] = torch.cat(batched_edges, dim=1) 
        
        masks = []
        for b in flat_batch:
            L = b["res_type"].shape[0]
            pad_len = max_L - L
            masks.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
        collated["mask"] = torch.stack(masks).bool()
        
        return collated
    

class ZarrTrajectoriesDataset(Dataset):
    ''' 
    Precomputed zarr files accession for CA or BB coordinates, 
    supporting unified mdCATH and ATLAS loading.
    '''
    def __init__(self, featuriser, mdcath_zarr_path, use_atlas=False, atlas_zarr_path=None,
                 choose_temps=None, max_domains=None, window_size=256, samples_per_traj=1,
                 crop=True, crop_size=384, native_aligned=True, coords_type='ca',
                 atlas_stride=100, rmsf_prior_path=None, verbose=False,
                 randomize_windows=False,
                 per_residue_dc_baselines=None):
        Dataset.__init__(self)

        self.featuriser = featuriser
        self.mdcath_zarr_path = mdcath_zarr_path
        self.use_atlas = use_atlas
        self.atlas_zarr_path = atlas_zarr_path
        self.verbose = verbose

        self.window_size = window_size
        self.samples_per_traj = samples_per_traj
        self.randomize_windows = bool(randomize_windows)
        self.crop = crop
        self.crop_size = crop_size
        self.native_aligned = native_aligned
        self.atlas_stride = atlas_stride
        self._dssp_cache = {}

        # Optional per-(protein, temp) per-residue DC baseline table. Keys are
        # strings ``f"{key}|{temp}"`` mapping to ``(L_full, coord_channels)``
        # float tensors in spectrum-space DC units. Loaded by the training
        # script from the conditioned freq-scale payload; ``None`` disables
        # the per-residue path and residualisation falls back to bucket DC.
        self.per_residue_dc_baselines = per_residue_dc_baselines

        # Optional NMA RMSF prior sidecar, produced by
        # scripts/precompute_nma_tica.py. Maps domain_id -> dict with at least
        # "rmsf_unit" (L,) — per-residue unitless ANM RMSF. Loaded eagerly on
        # the main process; workers see it via fork. Graceful fallback: if
        # the path is None or the file is missing, the loader simply does not
        # surface rmsf_prior in batches and the model must not rely on it.
        self.rmsf_prior_dict = None
        if rmsf_prior_path is not None:
            if not os.path.exists(rmsf_prior_path):
                print(f"WARNING: rmsf_prior_path={rmsf_prior_path} not found; continuing without prior")
            else:
                try:
                    self.rmsf_prior_dict = torch.load(
                        rmsf_prior_path, map_location="cpu", weights_only=False
                    )
                    print(f"Loaded NMA RMSF prior sidecar: {rmsf_prior_path} "
                          f"({len(self.rmsf_prior_dict)} domains)")
                except Exception as e:
                    print(f"WARNING: failed to load {rmsf_prior_path}: {e}; continuing without prior")
                    self.rmsf_prior_dict = None
        
        self.coords_type = coords_type.lower()
        if self.coords_type not in ['ca', 'bb']:
            raise ValueError("Zarr precomputed dataloader only supports 'ca' or 'bb' coords_type.")

        # Both CA and BB enumerate via bb_coords_native_aligned — it's the only array
        # written by the current extraction scripts. CA mode slices [:, :, 1, :] at fetch time.
        target_keys = ["bb_coords_native_aligned/.zarray"]

        self.index_map = []

        # ---------------------------------------------------------
        # 1. mdCATH Zarr Initialization
        # ---------------------------------------------------------
        print(f"Initializing mdCATH ZARR backend: {self.mdcath_zarr_path} | Mode: {self.coords_type.upper()}")
        self.zarr_root_mdcath, self.zarr_store_mdcath = _open_zarr_root_or_store(
            self.mdcath_zarr_path, "mdCATH"
        )
        
        mdcath_reps = []
        for path in self.zarr_store_mdcath.keys():
            if any(k in path for k in target_keys):
                parts = path.split('/')
                if len(parts) >= 4:
                    domain, temp, rep = parts[0], parts[1], parts[2]
                    # Append source tag 'mdcath'
                    mdcath_reps.append((domain, temp, rep, 'mdcath'))
        
        # Apply filters exclusively to mdCATH
        if max_domains is not None:
            unique_domains = list(dict.fromkeys(r[0] for r in mdcath_reps))[:max_domains]
            domain_set = set(unique_domains)
            mdcath_reps = [r for r in mdcath_reps if r[0] in domain_set]

        if choose_temps is not None:
            temp_set = set(str(t) for t in choose_temps)
            mdcath_reps = [r for r in mdcath_reps if r[1] in temp_set]

        self.index_map.extend(sorted(mdcath_reps))
        
        # ---------------------------------------------------------
        # 2. ATLAS Zarr Initialization
        # ---------------------------------------------------------
        if self.use_atlas:
            if not self.atlas_zarr_path:
                raise ValueError("use_atlas is True but atlas_zarr_path is not provided.")
            
            print(f"Initializing ATLAS ZARR backend: {self.atlas_zarr_path}")
            self.zarr_root_atlas, self.zarr_store_atlas = _open_zarr_root_or_store(
                self.atlas_zarr_path, "ATLAS"
            )
            
            atlas_reps = []
            for path in self.zarr_store_atlas.keys():
                if any(k in path for k in target_keys):
                    parts = path.split('/')
                    if len(parts) >= 4:
                        domain, temp, rep = parts[0], parts[1], parts[2]
                        # Append source tag 'atlas'
                        atlas_reps.append((domain, temp, rep, 'atlas'))
            
            # Append to the unified index map
            self.index_map.extend(sorted(atlas_reps))
        
        print(f"Dataset Ready: {len(self.index_map)} unified trajectories dynamically mapped.")

    def __len__(self):
        return len(self.index_map)
    
    def _init_worker_state(self):
        if not hasattr(self, "_worker_initialized"):
            self._worker_initialized = True
            self.zarr_root_mdcath, self.zarr_store_mdcath = _open_zarr_root_or_store(
                self.mdcath_zarr_path, "mdCATH"
            )
            if self.use_atlas:
                self.zarr_root_atlas, self.zarr_store_atlas = _open_zarr_root_or_store(
                    self.atlas_zarr_path, "ATLAS"
                )

    def _compute_domain_dssp(self, domain_key, domain_grp):
        if domain_key in self._dssp_cache:
            return self._dssp_cache[domain_key]

        if "pdbProteinAtoms" not in domain_grp:
            return None

        try:
            pdb_raw = domain_grp["pdbProteinAtoms"][...]
            if hasattr(pdb_raw, "item"):
                pdb_raw = pdb_raw.item()
            pdb_str = pdb_raw.decode("utf-8") if isinstance(pdb_raw, bytes) else str(pdb_raw)

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp.write(pdb_str.encode("utf-8"))
                tmp.flush()
                tmp_name = tmp.name
            try:
                traj = md.load_pdb(tmp_name)
                dssp_full = compute_native_dssp_onehot(traj, traj.topology.n_residues).numpy()
            finally:
                os.unlink(tmp_name)
        except Exception:
            return None

        self._dssp_cache[domain_key] = dssp_full
        return dssp_full

    def __getitem__(self, idx):
        self._init_worker_state()
        if self.verbose:
            print(f"[Worker {os.getpid()}] fetching idx {idx}")
        try:
            return self._fetch_data(idx)
        except Exception as e:
            key, temp, rep, source = self.index_map[idx]
            print(f"WARNING: Corrupted data at {source} - {key}/{temp}/{rep}: {e}. Resampling...")
            new_idx = random.randint(0, len(self.index_map) - 1)
            return self.__getitem__(new_idx)

    def _fetch_data(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Unpack the 4-tuple to determine data source
        key, temp, rep, source = self.index_map[idx]
        
        # Route to correct Zarr root and set the stride dynamically
        if source == 'atlas':
            domain_grp = _open_group_from_root_or_store(self.zarr_root_atlas, self.zarr_store_atlas, key)
            stride = self.atlas_stride
        else:
            domain_grp = _open_group_from_root_or_store(self.zarr_root_mdcath, self.zarr_store_mdcath, key)
            stride = 1
            
        rep_grp = domain_grp[temp][rep]
        
        # Determine keys based on coords_type
        slice_ca_from_bb = False
        
        if self.coords_type == 'bb':
            native_key = "native_bb_coords"
            coords_key = "bb_coords_native_aligned"
        else:
            # CA: always slice from BB — ca_coords_native_aligned is from an old pipeline
            # and not reliably present. bb_coords_native_aligned is the canonical source.
            native_key = "native_bb_coords"
            coords_key = "bb_coords_native_aligned"
            slice_ca_from_bb = True
        
        # Load Global Sequence / Native Features
        native_coords_full = domain_grp[native_key][...]
        res_type_full = domain_grp["res_type"][...]
        dssp_full = domain_grp["dssp"][...] if "dssp" in domain_grp else None
        if dssp_full is None:
            dssp_full = self._compute_domain_dssp(key, domain_grp)
        n_res = res_type_full.shape[0]
        
        native_angles_full = domain_grp["native_angles"][...] if "native_angles" in domain_grp else None
        torsion_mask_full = domain_grp["torsion_mask"][...] if "torsion_mask" in domain_grp else None

        # Load Trajectory Metadata
        coords_zarr = rep_grp[coords_key]
        angles_zarr = rep_grp.get("traj_angles", None)
        
        # Calculate strided time boundaries
        total_frames = coords_zarr.shape[0]
        effective_total_frames = total_frames // stride

        sub_batch = []
        for i in range(self.samples_per_traj):
            if self.randomize_windows:
                seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
            else:
                seed = stable_seed(idx, worker_id, rank, i)
            sub_rng = np.random.default_rng(seed)
            
            start_t, end_t = self.featuriser.get_time_window(effective_total_frames, sub_rng)
            real_start_t = start_t * stride
            real_end_t = end_t * stride

            if self.crop and n_res > self.crop_size:
                start_res = sub_rng.integers(0, n_res - self.crop_size + 1)
                end_res = start_res + self.crop_size
            else:
                start_res = 0
                end_res = n_res

            # Fetch via strided slice
            coords_window = torch.from_numpy(
                coords_zarr[real_start_t : real_end_t : stride, start_res:end_res]
            ).float()
            
            native_coords = torch.from_numpy(native_coords_full[start_res:end_res]).float()
            
            # THE ACTUAL CA SLICING HAPPENS HERE
            if slice_ca_from_bb:
                coords_window = coords_window[:, :, 1, :]  # [T, L, 4, 3] -> [T, L, 3]
                native_coords = native_coords[:, 1, :]     # [L, 4, 3] -> [L, 3]

            res_type = torch.from_numpy(res_type_full[start_res:end_res]).long()
            if dssp_full is not None:
                dssp = dssp_to_onehot(dssp_full[start_res:end_res], n_res=end_res - start_res)
            else:
                dssp = torch.zeros(end_res - start_res, len(DSSP_STATES), dtype=torch.float32)
                dssp[:, -1] = 1.0

            angles_window, native_angles, torsion_mask = None, None, None
            if self.featuriser.include_angles and angles_zarr is not None:
                angles_window = torch.from_numpy(
                    angles_zarr[real_start_t : real_end_t : stride, start_res:end_res]
                ).float()
                native_angles = torch.from_numpy(native_angles_full[start_res:end_res]).float()
                torsion_mask = torch.from_numpy(torsion_mask_full[start_res:end_res]).float()

            feats = self.featuriser.process(
                coords=coords_window,
                native_coords=native_coords,
                res_type=res_type,
                dssp=dssp,
                angles=angles_window,
                native_angles=native_angles,
                torsion_mask=torsion_mask,
            )

            if feats is not None:
                feats["temp"] = int(temp)
                feats["dataset_idx"] = idx
                feats["win_pos"] = start_t / max(effective_total_frames - 1, 1)
                feats["residue_idx"] = torch.arange(start_res, end_res, dtype=torch.long)

                # Optional NMA RMSF prior: slice the per-domain full-length
                # prior by the same residue crop used for native_coords.
                # Fails quietly if the domain is missing from the sidecar —
                # the model must then not be configured to require it.
                if self.rmsf_prior_dict is not None and key in self.rmsf_prior_dict:
                    rmsf_full = self.rmsf_prior_dict[key].get("rmsf_unit", None)
                    if rmsf_full is not None:
                        rmsf_full = torch.as_tensor(rmsf_full, dtype=torch.float32)
                        feats["rmsf_prior"] = rmsf_full[start_res:end_res].contiguous()

                if self.per_residue_dc_baselines is not None and source == "mdcath":
                    kt = f"{key}|{int(temp)}"
                    dc_row = self.per_residue_dc_baselines.get(kt)
                    if dc_row is not None:
                        dc_row = dc_row[start_res:end_res]
                        feats["dc_baseline_per_res"] = dc_row.float().contiguous()

                sub_batch.append(feats)

        return sub_batch

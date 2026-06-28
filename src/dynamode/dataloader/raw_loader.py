import os
import torch
from torch.utils.data import Dataset
import mdtraj as md
import tempfile
import numpy as np
import h5py
import hashlib
from collections import OrderedDict
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download
import torch.distributed as dist
import zarr
import tempfile
from functools import lru_cache
import numcodecs
numcodecs.blosc.use_threads = False

from dynamode.dataloader import *



def stable_seed(*args):
    s = "_".join(map(str, args))
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2**32


class TrajectoriesDataset(Dataset):
    repo_id = "compsciencelab/mdCATH"

    def __init__(self, featuriser, aligner, mdcath_path=None, atlas_path=None, choose_temps=None, 
                 max_domains=None, window_size=256, samples_per_traj=1, device='cpu', 
                 crop=True, crop_size=384, offline_mode=False, use_atlas=False, max_alignment_iters=2,
                 use_zarr=False, zarr_path=None, native_aligned=True, coords_type='ca', atlas_stride=100,
                 randomize_windows=False):

        Dataset.__init__(self)

        self.featuriser = featuriser
        self.aligner = aligner
        self.max_alignment_iters = max_alignment_iters 
        self.device = device
        self.native_aligned = native_aligned
        self.atlas_stride = atlas_stride

        self.coords_type = coords_type.lower()
        if self.coords_type not in ['ca', 'bb', 'all']:
            raise ValueError("coords_type must be 'ca', 'bb', or 'all'")
        
        self.mdcath_path = mdcath_path
        self.use_atlas = use_atlas
        if use_atlas:
            if atlas_path:
                self.atlas_path = atlas_path
            else:
                self.atlas_path = os.path.join(os.path.dirname(os.path.dirname(mdcath_path)), "ATLAS")
                print(f"Using ATLAS but path not provided so assuming above mdCATH path: {self.atlas_path}")

        self.window_size = window_size
        self.samples_per_traj = samples_per_traj 
        self.randomize_windows = bool(randomize_windows)
        self.crop = crop
        self.crop_size = crop_size 

        # Determine offline mode
        self.local_files_only = offline_mode
        if self.local_files_only:
            print(f"OFFLINE MODE: Loading ONLY files present locally.")

        # ZARR vs HDF5 Initialization
        # ---------------------------
        self.use_zarr = use_zarr
        if self.use_zarr:
            self.zarr_path = zarr_path if zarr_path else os.path.join(self.mdcath_path, "mdcath_unified.zarr")
            print(f"Using ZARR backend: {self.zarr_path}")
            
            # Open consolidated metadata for blazing fast init
            self.zarr_root = zarr.open_consolidated(self.zarr_path, mode='r')
            
            self.all_keys = list(self.zarr_root.keys())
            first_k = self.all_keys[0]
            # Exclude the PDB metadata dataset from the temp list
            self.all_temps = [k for k in self.zarr_root[first_k].keys() if k != "pdbProteinAtoms"]
            self.all_reps = list(self.zarr_root[first_k][self.all_temps[0]].keys())
            
        else:
            print("Using HDF5 backend.")
            # Define expected locations
            source_filename = "mdcath_source.h5"
            possible_paths = [
                os.path.join(self.mdcath_path, source_filename),          
                os.path.join(self.mdcath_path, 'data', source_filename)   
            ]
            source_file = None
            for p in possible_paths:
                if os.path.exists(p):
                    source_file = p
                    break

            if source_file is None:
                source_file = hf_hub_download(
                    repo_id=self.repo_id, 
                    filename=source_filename, 
                    repo_type="dataset",
                    local_dir=self.mdcath_path
                )
            
            with h5py.File(source_file, "r") as f:
                self.all_keys = list(f.keys())
                first_k = self.all_keys[0]
                self.all_temps = list(f[first_k].keys())
                self.all_reps = list(f[first_k][self.all_temps[0]].keys())

        # Filter Index Map
        # ----------------
        self.index_map = []
        temps_to_use = choose_temps if choose_temps else self.all_temps
        keys_to_use = self.all_keys[:max_domains] if max_domains else self.all_keys

        valid_keys = []
        missing = []
        
        if self.use_zarr:
            # If using Zarr, all keys in the store are inherently valid and local
            valid_keys = keys_to_use
        elif self.local_files_only:
            for i, k in enumerate(keys_to_use):
                fname = f"mdcath_dataset_{k}.h5"
                if os.path.exists(os.path.join(self.mdcath_path, fname)):
                    valid_keys.append(k)
                else:
                    print(f"missing file name {fname}")
                    missing.append(i)
            print(f"Offline Filter: {len(valid_keys)} / {len(keys_to_use)} domains found locally.")
            keys_to_use = valid_keys
            
        if len(keys_to_use) == 0:
            raise RuntimeError(f"No valid mdCATH files found in {self.mdcath_path} with offline_mode=True")

        # Final INDEX MAP
        # ---------------
        for key in keys_to_use:
            for temp in temps_to_use:
                for rep in self.all_reps:
                    self.index_map.append((key, temp, rep, 'mdcath'))
        
        # ATLAS
        self.atlas_ids = []
        if use_atlas:
            with open(os.path.join(self.atlas_path, "atlas.txt"), "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped: 
                        self.atlas_ids.append(stripped)
                        self.index_map.append((stripped, "300", "0", 'atlas'))
        
        print(f"Dataset Ready: {len(self.index_map)} trajectories.")

    def __len__(self):
        return len(self.index_map)
    
    def _init_worker_state(self):
        if not hasattr(self, "_worker_initialized"):
            self._worker_initialized = True
            self._worker_topology_cache = OrderedDict()
            self._worker_native_xyz_cache = OrderedDict()
            self._worker_dssp_cache = OrderedDict()
            #self._worker_pdb_cache = {}
            #self._worker_native_traj_cache = {}
            self._cache_maxsize = 2000
            self.zarr_root = zarr.open_consolidated(self.zarr_path, mode='r') if self.use_zarr else None
    
    def _cache_put(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)

        if len(cache) > self._cache_maxsize:
            cache.popitem(last=False)  # evict LRU

    @staticmethod
    @lru_cache(maxsize=100)
    def _get_pdb_object(pdb_str):
        tf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            tf.write(pdb_str.encode("utf-8"))
            tf.close()
            return md.load_pdb(tf.name)
        finally:
            os.unlink(tf.name)
    
    def get_topology(self, key):
        # Per-worker cache
        if key in self._worker_topology_cache:
            return self._worker_topology_cache[key]

        pdb_str = str(self.zarr_root[key]["pdbProteinAtoms"][...].item())

        # Must use real file (mdtraj requirement)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_str.encode("utf-8"))
            tmp.close()
            top = md.load_pdb(tmp.name).topology
            os.unlink(tmp.name)

        self._cache_put(self._worker_topology_cache, key, top)
        return top
    
    def _get_atom_indices(self, topology, start_res, end_res):
        '''
        Extracts the desired atom indices based on self.coords_type.

        When coords_type='ca' but the featurizer requests angles, backbone atoms
        (N, CA, C, O) are loaded instead of CA-only so that mdtraj can compute
        phi/psi in the featurizer.  The featurizer selects CA from the backbone
        topology via topology.select('name CA'), so its output is still CA coords.
        '''
        # Fast path if we want everything
        if self.coords_type == 'all' and start_res == 0 and end_res == topology.n_residues:
            return None

        # Angles require backbone atoms even in CA mode
        needs_backbone = (
            self.coords_type == 'ca'
            and getattr(self.featuriser, 'include_angles', False)
        )
        effective_type = 'bb' if needs_backbone else self.coords_type

        atom_indices = []
        for r_idx in range(start_res, end_res):
            res = topology.residue(r_idx)

            if effective_type == 'all':
                atom_indices.extend([a.index for a in res.atoms])

            elif effective_type == 'ca':
                ca_idx = next((a.index for a in res.atoms if a.name == 'CA'), -1)
                if ca_idx != -1:
                    atom_indices.append(ca_idx)

            elif effective_type == 'bb':
                atom_dict = {a.name: a.index for a in res.atoms}
                o_idx = atom_dict.get('O', atom_dict.get('OXT', atom_dict.get('O1', atom_dict.get('OT1', -1))))
                n_idx = atom_dict.get('N', -1)
                ca_idx = atom_dict.get('CA', -1)
                c_idx = atom_dict.get('C', -1)

                for idx in [n_idx, ca_idx, c_idx, o_idx]:
                    if idx != -1:
                        atom_indices.append(idx)

        return atom_indices

    def _process_trajectory(self, key, topology, native_xyz, total_frames, temp, load_window_fn, idx, worker_id=0, rank=0, dssp_full=None):
        sub_batch = []

        for i in range(self.samples_per_traj):
            if self.randomize_windows:
                seed = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
            else:
                seed = stable_seed(idx, worker_id, rank, i)
            sub_rng = np.random.default_rng(seed)
            start_t, end_t = self.featuriser.get_time_window(total_frames, sub_rng)

            n_res = topology.n_residues
            if self.crop and n_res > self.crop_size:
                start_res = sub_rng.integers(0, n_res - self.crop_size + 1)
                end_res = start_res + self.crop_size
            else:
                start_res = 0
                end_res = n_res
                
            # Extract only the specified atoms
            atom_indices = self._get_atom_indices(topology, start_res, end_res)

            traj_chunk_xyz = load_window_fn(start_t, end_t, atom_indices) 
            
            # MAJOR BUG FIX: we were previously creating zeros coords for the native structure when using the Zarr backend, which caused all features to be NaN. Now we ensure the native trajectory is properly created from the topology for both backends.
            # native_coords = np.zeros((1, topology.n_atoms, 3), dtype=np.float32)
            # native_traj = md.Trajectory(native_coords, topology)

            # cache option
            #if key in self._worker_native_traj_cache:
            #    native_traj = self._worker_native_traj_cache[key]
            #else:
            native_traj = md.Trajectory(native_xyz, topology)
            #    self._worker_native_traj_cache[key] = native_traj

            if atom_indices is not None:
                cropped_top = topology.subset(atom_indices)
                traj_chunk = md.Trajectory(traj_chunk_xyz, cropped_top)
                native_traj_crop = native_traj.atom_slice(atom_indices)
            else:
                traj_chunk = md.Trajectory(traj_chunk_xyz, topology)
                native_traj_crop = native_traj

            # NEW ALIGNMENT TARGET 06.04.26
            # Align trajectory to native CA coords (not frame 0 or window mean).
            # native_traj_crop.xyz is in nm; convert to Å and select CA before passing.
            if self.native_aligned:            
                # Even if we load 'bb' or 'all', Kabsch rotation should be calculated using the CA core
                ca_sel = native_traj_crop.topology.select('name CA')
                native_ca_ang = torch.tensor(
                    native_traj_crop.xyz[0, ca_sel, :], dtype=torch.float32
                ) * 10.0  # (L, 3) Å
                align_target = native_ca_ang
            else: align_target = None
            if dssp_full is not None:
                dssp_crop = dssp_to_onehot(dssp_full[start_res:end_res], n_res=end_res - start_res)
            else:
                dssp_crop = None
            feats = self.featuriser.process(
                traj_chunk, native_traj_crop, align_target=align_target, rng=sub_rng, dssp=dssp_crop
            )

            if feats is not None:
                feats["temp"] = int(temp)
                feats["dataset_idx"] = idx
                feats["win_pos"] = start_t / max(total_frames - 1, 1)
                feats["residue_idx"] = torch.arange(start_res, end_res, dtype=torch.long)
                sub_batch.append(feats)

        return sub_batch

    def __getitem__(self, idx, verbose=False):
        self._init_worker_state()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        for _ in range(10): # 10 attempts
            key, temp, rep, source = self.index_map[idx]
            # print(f"[Worker PID: {os.getpid()}] Attempting to load key: {key}", flush=True)

            # ATLAS
            # --------------------
            if source == 'atlas':
                # Flat layout: {atlas_path}/{key}_topo.pdb
                pdb_file = os.path.join(self.atlas_path, f"{key}_topo.pdb")
                xtc_file = os.path.join(self.atlas_path, f"{key}_sliced.xtc")

                # Subdirectory layout (download_atlas.py): {atlas_path}/{key}/{key}_topo.pdb
                if not os.path.exists(pdb_file):
                    sub_pdb = os.path.join(self.atlas_path, key, f"{key}_topo.pdb")
                    if os.path.exists(sub_pdb):
                        pdb_file = sub_pdb
                if not os.path.exists(xtc_file):
                    subdir = os.path.join(self.atlas_path, key)
                    if os.path.isdir(subdir):
                        # Per-replica XTCs: {key}_{repeat}_sliced.xtc
                        xtc_candidates = sorted(
                            f for f in os.listdir(subdir)
                            if f.startswith(f"{key}_") and f.endswith("_sliced.xtc")
                        )
                        if xtc_candidates:
                            xtc_file = os.path.join(subdir, xtc_candidates[0])

                if not os.path.exists(pdb_file) or not os.path.exists(xtc_file):
                    print(f"[Worker {os.getpid()}] ATLAS files not found for {key}: "
                          f"pdb={pdb_file} (exists={os.path.exists(pdb_file)}), "
                          f"xtc={xtc_file} (exists={os.path.exists(xtc_file)})", flush=True)
                    return []

                pdb_obj = md.load(pdb_file)
                if pdb_obj is None:
                    return []
                atlas_dssp = compute_native_dssp_onehot(pdb_obj, pdb_obj.topology.n_residues)

                with md.formats.XTCTrajectoryFile(xtc_file) as f_xtc:
                    # Scale down the total frames by the stride for the featurizer's RNG
                    effective_total_frames = len(f_xtc) // self.atlas_stride

                def atlas_loader(start, end, atom_indices=None):
                    chunk_size = end - start
                    if chunk_size < 1:
                        return np.array([])
                    
                    # Convert the featurizer's downsampled start back to the raw file's real frame index
                    real_skip = start * self.atlas_stride
                    
                    # Pass stride directly to iterload. It will fetch `chunk_size` frames, 
                    # separated by `stride` frames, starting at `real_skip`.
                    iterator = md.iterload(
                        xtc_file, 
                        top=pdb_file, 
                        skip=real_skip, 
                        stride=self.atlas_stride, 
                        chunk=chunk_size
                    )
                    chunk = next(iterator)
                    del iterator 
                    
                    xyz = chunk.xyz
                    if atom_indices is not None:
                        xyz = xyz[:, atom_indices, :]
                    return xyz

                result = self._process_trajectory(
                    key, pdb_obj.topology, pdb_obj.xyz, effective_total_frames, temp, atlas_loader, idx, worker_id, rank,
                    dssp_full=atlas_dssp,
                )
                if len(result) > 0:
                    return result

            # mdCATH ZARR branch
            # --------------------
            elif source == 'mdcath':
                if self.use_zarr:
                    coords_array = self.zarr_root[key][temp][rep]["coords"]
                    native_xyz = self.zarr_root[key]["native_coords"][...]
                    dssp_full = self.zarr_root[key]["dssp"][...] if "dssp" in self.zarr_root[key] else None
                    total_frames = coords_array.shape[0]

                    try:
                        topology = self.get_topology(key)
                    except Exception as e:
                        print(f"WARNING: Failed to load topology for {key}. Skipping. ({e})")
                        return []
                    if topology is None:
                        print(f"WARNING: Failed to load topology for {key}. Skipping trajectory.")
                        return []

                    if dssp_full is None:
                        if key in self._worker_dssp_cache:
                            dssp_full = self._worker_dssp_cache[key]
                            self._worker_dssp_cache.move_to_end(key)
                        else:
                            try:
                                native_traj = md.Trajectory(native_xyz, topology)
                                dssp_full = compute_native_dssp_onehot(
                                    native_traj, topology.n_residues
                                ).numpy()
                                self._cache_put(self._worker_dssp_cache, key, dssp_full)
                            except Exception:
                                dssp_full = None

                    def zarr_loader(start, end, atom_indices=None):
                        try:
                            # Direct slicing is extremely fast with Zarr
                            xyz_ang = coords_array[start:end]
                        except Exception as e:
                            # Force the domain info into the actual PyTorch error traceback
                            crash_msg = (f"\n\n[!!!] CORRUPTED ZARR CHUNK DETECTED [!!!]\n"
                                        f"Domain: {key} | Temp: {temp} | Rep: {rep}\n"
                                        f"Delete it with: rm -rf {self.zarr_path}/{key}\n\n")
                            raise RuntimeError(crash_msg) from e
                            
                        if atom_indices is not None:
                            xyz_ang = xyz_ang[:, atom_indices, :]
                        return xyz_ang / 10.0

                    result = self._process_trajectory(
                        key, topology, native_xyz, total_frames, temp, zarr_loader, idx, worker_id, rank,
                        dssp_full=dssp_full,
                    )
                    if len(result) > 0:
                        # print(f"[Worker PID: {os.getpid()}] Loaded key: {key}", flush=True)
                        return result
                
                else:
                    # mdCATH HDF5 branch
                    # --------------------
                    fname = f"mdcath_dataset_{key}.h5"
                    p1 = os.path.join(self.mdcath_path, fname)
                    p2 = os.path.join(self.mdcath_path, 'data', fname)
                    h5path = p1 if os.path.exists(p1) else p2

                    if not os.path.exists(h5path):
                        if self.local_files_only:
                            raise FileNotFoundError(h5path)
                        h5path = hf_hub_download(
                            "compsciencelab/mdCATH",
                            fname,
                            subfolder='data',
                            local_dir=self.mdcath_path
                        )
                
                    # Old topology cache - UNSAFE TO OPEN AND CLOSE H5file with num_workers > 0 due to HDF5 thread safety issues, so we now load topology on-the-fly with LRU cache in get_topology() method
                    # with h5py.File(h5path, 'r') as f:
                    #     total_frames = f[key][temp][rep]["coords"].shape[0]
                    # topology = self.topology_cache.get(key)
                    # if topology is None:
                    #     print(f"WARNING: Failed to load topology for {key}. Skipping trajectory.")
                    #     return []

                    # SAFE SOLUTION IS TO LOAD TOPOLOGY ON-THE-FLY INSIDE THE WORKER WITH ITS OWN FILE HANDLE, AND CACHE IN A PER-WORKER LRU CACHE. THIS AVOIDS ALL HDF5 THREAD SAFETY ISSUES AND ALSO MEANS WE ONLY LOAD TOPOLOGIES ACTUALLY NEEDED BY THE WORKER.
                    with h5py.File(h5path, 'r') as f:
                        ds = f[key][temp][rep]["coords"]
                        total_frames = ds.shape[0]
                        dssp_full = f[key]["dssp"][()] if "dssp" in f[key] else None

                        # load topology inside worker and cache in worker cache - 
                        #pdb_str = f[key]["pdbProteinAtoms"][()].decode("utf-8")]
                        # Let's do worker cache instead
                        # if key in self._worker_pdb_cache:
                        #     pdb_str = self._worker_pdb_cache[key]
                        # else:
                        pdb_str = f[key]["pdbProteinAtoms"][()].decode("utf-8")
                        # self._worker_pdb_cache[key] = pdb_str

                        if key in self._worker_topology_cache:
                            topology = self._worker_topology_cache[key]
                            self._worker_topology_cache.move_to_end(key)
                            native_xyz = self._worker_native_xyz_cache[key]
                            self._worker_native_xyz_cache.move_to_end(key)
                        else:
                            tf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
                            try:
                                tf.write(pdb_str.encode("utf-8"))
                                tf.close()
                                pdb_obj = md.load_pdb(tf.name)
                            finally:
                                os.unlink(tf.name)

                            topology = pdb_obj.topology
                            native_xyz = pdb_obj.xyz.copy()

                            # self._worker_topology_cache[key] = topology
                            # self._worker_native_xyz_cache[key] = native_xyz
                            self._cache_put(self._worker_topology_cache, key, topology)
                            self._cache_put(self._worker_native_xyz_cache, key, native_xyz)

                        if dssp_full is None:
                            if key in self._worker_dssp_cache:
                                dssp_full = self._worker_dssp_cache[key]
                                self._worker_dssp_cache.move_to_end(key)
                            else:
                                try:
                                    native_traj = md.Trajectory(native_xyz, topology)
                                    dssp_full = compute_native_dssp_onehot(
                                        native_traj, topology.n_residues
                                    ).numpy()
                                    self._cache_put(self._worker_dssp_cache, key, dssp_full)
                                except Exception:
                                    dssp_full = None

                        def mdcath_loader(start, end, atom_indices=None):
                            xyz_ang = ds[start:end]
                            if atom_indices is not None:
                                xyz_ang = xyz_ang[:, atom_indices, :]
                            return xyz_ang / 10.0

                        result = self._process_trajectory(
                            key, topology, native_xyz, total_frames, temp, mdcath_loader, idx, worker_id, rank,
                            dssp_full=dssp_full,
                        )

                        if len(result) > 0:
                            return result
            
            print(f"[Worker PID: {os.getpid()}] Key {key} failed or missing. Resampling...", flush=True)
        raise RuntimeError(f"Failed to load sample for key={key}")

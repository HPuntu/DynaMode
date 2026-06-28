import torch
from torch.nn import functional as F
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation


DSSP_STATES = ("H", "B", "E", "G", "I", "T", "S", "C")
DSSP_MAP = {state: i for i, state in enumerate(DSSP_STATES)}


def dssp_to_onehot(raw_dssp, n_res: int | None = None) -> torch.Tensor:
    '''
    Canonicalise DSSP-like input to an (L, 8) float tensor.

    Accepts any of the following:
    - (L, 8) float/int scores or one-hot
    - (1, L, 8) scores
    - (L,) integer class ids in [0, 7]
    - (1, L) / (L,) string labels from mdtraj.compute_dssp
    '''
    if torch.is_tensor(raw_dssp):
        arr = raw_dssp.detach().cpu().numpy()
    else:
        arr = np.asarray(raw_dssp)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2 and arr.shape[-1] == len(DSSP_STATES) and np.issubdtype(arr.dtype, np.number):
        out = torch.as_tensor(arr, dtype=torch.float32)
        if n_res is not None and out.shape[0] != n_res:
            raise ValueError(f"DSSP length mismatch: expected {n_res}, got {out.shape[0]}")
        return out

    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 1:
        raise ValueError(f"Unsupported DSSP shape {tuple(arr.shape)}")

    if n_res is not None and arr.shape[0] != n_res:
        raise ValueError(f"DSSP length mismatch: expected {n_res}, got {arr.shape[0]}")

    if np.issubdtype(arr.dtype, np.number):
        idx = torch.as_tensor(arr, dtype=torch.long).clamp(min=0, max=len(DSSP_STATES) - 1)
        return F.one_hot(idx, num_classes=len(DSSP_STATES)).float()

    labels = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            token = x.decode("utf-8").strip()
        else:
            token = str(x).strip()
        token = token[0] if token else "C"
        labels.append(DSSP_MAP.get(token, DSSP_MAP["C"]))
    idx = torch.tensor(labels, dtype=torch.long)
    return F.one_hot(idx, num_classes=len(DSSP_STATES)).float()


def compute_native_dssp_onehot(traj: md.Trajectory, n_res: int) -> torch.Tensor:
    '''Compute native DSSP on the first frame and return (L, 8) float.'''
    try:
        dssp = md.compute_dssp(traj, simplified=False)
    except Exception:
        # Coil fallback is safer than returning NaNs and keeps conditioning shape-stable.
        dssp = np.full((1, n_res), "C", dtype="<U1")
    return dssp_to_onehot(dssp, n_res=n_res)


class Aligner:
    '''
    Class for aligning trajectories
    '''
    @staticmethod
    def _compute_kabsch_rotation(P, Q):
        '''
        Standard vectorized Kabsch.
        P, Q: [..., N, 3]  
        Returns: R [..., 3, 3]
        '''
        H = torch.matmul(P.transpose(-1, -2), Q)
        U, S, Vt = torch.linalg.svd(H) 
        d = torch.sign(torch.linalg.det(torch.matmul(U, Vt)))
        diag = torch.ones(H.shape[:-1], device=P.device)
        diag[..., 2] = d
        R = torch.matmul(torch.matmul(U, torch.diag_embed(diag)), Vt)
        return R
    
    def align(self, coords, ref_coords=None, core_fraction=0.5, max_iters=2, apply_coords=None):
        '''
        Performs iterative alignment to prevent flexible loops from causing 
        translation and rotation leakage into the stationary core.

        Can be used non-iteratively by specifying max_iters=1
        '''
        T, N, _ = coords.shape
        device = coords.device

        # Initialize reference. If None, start with Frame 0.
        # This will be updated to the window Mean in the loop.
        if ref_coords is None:
            ref_coords = coords[0]
        
        # Start Pass 1 with ALL residues as the 'core'
        core_indices = torch.arange(N, device=device)
        aligned_coords = coords.clone()
        
        for i in range(max_iters):
            # 1. Calculate centers of mass using ONLY the current core residues
            coords_core_mean = coords[:, core_indices, :].mean(dim=1, keepdim=True) # (T, 1, 3)
            
            # Use the Mean of the ALIGNED batch as the reference
            # for the next pass. This aligns the trajectory to its own average structure.
            #if i > 0: ref_structure = aligned_coords.mean(dim=0) # (N, 3)
            #else: ref_structure = ref_coords
            ref_structure = ref_coords

            ref_core_mean = ref_structure[core_indices, :].mean(dim=0, keepdim=True)
            
            # Center ALL atoms based on the core's center of mass
            coords_centered = coords - coords_core_mean
            ref_centered = ref_structure - ref_core_mean
            
            # 2. Slice out just the core for calculating the rotation matrix
            P_core = coords_centered[:, core_indices, :]
            Q_core = ref_centered[core_indices, :].unsqueeze(0).expand(T, -1, -1)
            
            # 3. Compute rotation using Kabsch on the core
            R = self._compute_kabsch_rotation(P_core, Q_core)
            
            # 4. Apply rotation to ALL coordinates
            aligned_coords = torch.matmul(coords_centered, R)
            
            if i < max_iters - 1:
                # Calculate TRUE RMSF (Variance relative to its own mean)
                window_mean = aligned_coords.mean(dim=0, keepdim=True) # (1, N, 3)
                sq_diff = (aligned_coords - window_mean)**2
                rmsf = sq_diff.sum(dim=-1).mean(dim=0) # (N,)
                
                # Keep the most stable fraction of residues for the next iteration
                k = max(3, int(N * core_fraction))
                _, core_indices = torch.topk(rmsf, k, largest=False)
                
        # aligned_coords is already centred on the stable core COM (coords were centred
        # on core_indices COM before rotation in the final pass). Return core_indices
        # so callers can centre native_coords on the same residues.

        # Apply the final rotation to the full backbone
        if apply_coords is not None:
            # apply_coords is [T, L, 4, 3]. coords_core_mean is [T, 1, 3].
            # Unsqueeze to [T, 1, 1, 3] for broadcasting across the 4 atoms
            apply_centered = apply_coords - coords_core_mean.unsqueeze(-2)
            
            # Flatten [T, L, 4, 3] -> [T, L*4, 3] to match R's batch dimension [T, 3, 3]
            T_dim, L_dim, A_dim, _ = apply_centered.shape
            apply_flat = apply_centered.view(T_dim, L_dim * A_dim, 3)
            
            # Apply rotation
            aligned_apply_flat = torch.matmul(apply_flat, R)
            
            # Reshape back to [T, L, 4, 3]
            aligned_apply = aligned_apply_flat.view(T_dim, L_dim, A_dim, 3)
            
            return aligned_coords, core_indices, aligned_apply
        
        return aligned_coords, core_indices


class Featurizer:
    '''
    Base class for different protocol featurisers to inherit from.
    Includes functionality for SE(3) rigid body transformation
    '''
    AA_MAP = {
        'ALA':0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7,
        'HIS':8, 'ILE':9, 'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14,
        'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19, 'UNK':20
    }

    def __init__(self, aligner, device='cpu', max_alignment_iters=2):
        self.aligner = aligner
        self.device = device
        self.max_alignment_iters = max_alignment_iters

    def _get_sequence_onehot(self, topology):
        indices = []
        for res in topology.residues:
            name = res.name if res.name in self.AA_MAP else 'UNK'
            indices.append(self.AA_MAP[name])
        return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=21).float()

    @staticmethod
    def _get_random_rotation():
        return torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
    
    @staticmethod
    def _construct_rigid_bodies(xyz_n, xyz_ca, xyz_c):
        # Standard Gram-Schmidt orthogonalization
        T = xyz_ca 
        v1 = xyz_n - xyz_ca
        v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-6)
        v2 = xyz_c - xyz_ca
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-6)
        u = torch.cross(v1, v2, dim=-1)
        u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-6)
        v = torch.cross(u, v1, dim=-1)
        R = torch.stack([v1, v, u], dim=-1)
        return R, T
    
    @staticmethod
    def _so3_log_map(R_current, R_next):
        '''Computes tangent vector along the geodesic of the so(3) lie algebra'''
        R_rel = torch.matmul(R_current.transpose(-1, -2), R_next)
        
        # 1. Compute theta
        tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        
        # 2. Extract unscaled axis vector directly
        R_diff = R_rel - R_rel.transpose(-1, -2)
        axis_unnorm = torch.stack([
            R_diff[..., 2, 1], 
            R_diff[..., 0, 2], 
            R_diff[..., 1, 0]
        ], dim=-1)
        
        # 3. Scale safely without epsilon bias
        # Limit as theta -> 0 of (theta / 2*sin(theta)) is 0.5
        safe_denom = torch.where(
            theta < 1e-4, 
            torch.ones_like(theta) * 2.0, 
            2.0 * torch.sin(theta) / theta
        )
        
        return axis_unnorm / safe_denom.unsqueeze(-1)

    def get_time_window(self, total_frames):
        '''PLACEHOLDER - Returns (start_t, end_t) specifying the chunk the dataset should load'''
        raise NotImplementedError

    def process(self, traj_chunk, native_traj):
        '''PLACEHOLDER - Extracts features from the cropped trajectories'''
        raise NotImplementedError

    def collate_fn(self, batch):
        '''PLACEHOLDER - Pads and batches the specific dictionary outputs of this featurizer'''
        raise NotImplementedError

    @staticmethod
    def _pad_tensor(t, pad_len, spatial_dim):
        '''
        Universal dynamic padder
        spatial_dim: The index of the dimension representing Sequence Length (L)
        '''
        if pad_len == 0: return t
        
        # F.pad expects pairs of (left, right) padding from the LAST dim to the FIRST.
        pads = [0] * (2 * t.ndim)
        
        # Calculate the index in the padding tuple for the RIGHT pad of our spatial_dim
        dist_from_last = t.ndim - 1 - spatial_dim
        right_pad_idx = dist_from_last * 2 + 1
        
        pads[right_pad_idx] = pad_len
        return F.pad(t, tuple(pads))

  
class FeaturizerWindow(Featurizer):
    '''
    For Spectral Volume Diffusion 
    '''

    def __init__(self, aligner, window_size=256, device='cpu', max_alignment_iters=2, include_angles=True, coords_type="ca"):
        super().__init__(aligner, device, max_alignment_iters=max_alignment_iters)
        self.window_size = window_size
        self.include_angles = include_angles # if False will just provide coords
        self.coords_type = coords_type

    def get_backbone_indices(self, topology):
        """Helper to get N, CA, C, O indices for the full backbone."""
        bb_indices = []
        for res in topology.residues:
            atom_dict = {a.name: a.index for a in res.atoms}
            o_idx = atom_dict.get('O', atom_dict.get('OXT', atom_dict.get('O1', atom_dict.get('OT1', -1))))
            bb_indices.append([
                atom_dict.get('N', -1), 
                atom_dict.get('CA', -1), 
                atom_dict.get('C', -1), 
                o_idx
            ])
        return torch.tensor(bb_indices, dtype=torch.long)

    def get_time_window(self, total_frames, rng=None):
        if total_frames <= self.window_size: 
            return 0, int(total_frames)
        start = rng.integers(0, total_frames - self.window_size)
        return start, start + self.window_size

    def _get_angles(self, traj, n_res):
        try:
            _, phi = md.compute_phi(traj)
            _, psi = md.compute_psi(traj)
        except Exception:
            print("WARNING: Failed to compute angles with mdtraj")
            return None

        # mdtraj returns fewer dihedrals than n_res-1 when the chain has breaks or
        # non-standard residues (can also return 0 columns silently).
        if phi.shape[1] != n_res - 1 or psi.shape[1] != n_res - 1:
            print(f"WARNING: Unexpected phi/psi count (phi={phi.shape[1]}, psi={psi.shape[1]}, n_res={n_res}). Skipping angles.")
            return None

        phi_pad = torch.zeros((traj.n_frames, n_res))
        psi_pad = torch.zeros((traj.n_frames, n_res))

        phi_pad[:, 1:] = torch.from_numpy(phi)
        psi_pad[:, :-1] = torch.from_numpy(psi)

        return torch.stack([
            torch.sin(phi_pad), torch.cos(phi_pad),
            torch.sin(psi_pad), torch.cos(psi_pad)
        ], dim=-1)
    
    def _get_ca_static_edges(self, n_res):
        '''CA-CA static edges for thermal jitter upscaler GNN - not use for spectral volume diffusion'''
        src, dst = [], []
        # Connect i to i+1 (bonds), i+2 (angles), and i+3 (dihedrals)
        for hop in [1, 2, 3]:
            for i in range(n_res - hop):
                src.extend([i, i + hop])
                dst.extend([i + hop, i]) # Make edges bidirectional for message passing
                
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
            
        return torch.tensor([src, dst], dtype=torch.long)

    def process(self, traj_window, native_traj, align_target=None, rng=None, dssp=None):
        # 1. Setup Data
        xyz_angstrom = torch.tensor(traj_window.xyz, dtype=torch.float32) * 10.0
        T_frames = xyz_angstrom.shape[0]
        
        topology = traj_window.topology
        n_res = topology.n_residues
        ca_indices = topology.select('name CA')

        # 2. Align trajectory to native; align() centres each frame on the stable
        #    core COM and returns the core indices used in the final pass.
        #    works for bb and ca only atoms
        xyz_ca = xyz_angstrom[:, ca_indices, :].float()
        ref_coords = align_target.to(xyz_ca.device) if align_target is not None else None

        if self.coords_type == "bb":
            bb_indices = self.get_backbone_indices(topology)
            xyz_bb = xyz_angstrom[:, bb_indices, :].float()
            
            # Pass full backbone to apply_coords
            ca_aligned, core_idx, aligned_coords = self.aligner.align(
                xyz_ca, 
                ref_coords=ref_coords, 
                core_fraction=0.5, 
                max_iters=self.max_alignment_iters,
                apply_coords=xyz_bb  # <--- Apply CA rotation to full BB
            )
            # aligned_coords is [T, L, 4, 3] -> flatten to [T, L, 12]
            T_f, L_f, A, xyz = aligned_coords.shape
            aligned_coords = aligned_coords.reshape(T_f, L_f, A * xyz)

            # Process native backbone
            native_angstrom = torch.tensor(native_traj.xyz, dtype=torch.float32) * 10.0
            native_bb = native_angstrom[:, bb_indices, :].float().squeeze(0) # (L, 4, 3)
            native_ca = native_bb[:, 1, :]
            native_core_mean = native_ca[core_idx].mean(dim=0, keepdim=True)
            native_coords = (native_bb - native_core_mean.unsqueeze(-2)).reshape(L_f, A * xyz)  # (L, 12)
            
        else: # "ca" mode
            aligned_coords, core_idx = self.aligner.align(
                xyz_ca, 
                ref_coords=ref_coords, 
                core_fraction=0.5, 
                max_iters=self.max_alignment_iters
            )
            # aligned_coords is [T, L, 3] — keep flat

            # Process native CA
            native_angstrom = torch.tensor(native_traj.xyz, dtype=torch.float32) * 10.0
            native_ca = native_angstrom[:, ca_indices, :].float().squeeze(0)
            native_core_mean = native_ca[core_idx].mean(dim=0, keepdim=True)
            native_coords = native_ca - native_core_mean  # (L, 3)

        # 4. Process Sequence & Static Edges
        res_type = self._get_sequence_onehot(topology)#.to(self.device)
        if dssp is None:
            dssp = compute_native_dssp_onehot(native_traj, n_res)
        else:
            dssp = dssp_to_onehot(dssp, n_res=n_res)
        static_edges = self._get_ca_static_edges(n_res)

        # 5. Handle Time Padding for Coords
        time_mask = torch.ones(self.window_size)
        if T_frames < self.window_size:
            diff = self.window_size - T_frames
            time_mask = torch.cat([torch.ones(T_frames), torch.zeros(diff)])
            coords_pad = aligned_coords[-1].unsqueeze(0).repeat(diff, *([1] * (aligned_coords.dim() - 1)))
            aligned_coords = torch.cat([aligned_coords, coords_pad], dim=0)

        # 6. Build Base Dictionary
        out = {
            "coords": aligned_coords,               
            "native_coords": native_coords, 
            "res_type": res_type,
            "dssp": dssp,
            "time_mask": time_mask,         
            "static_edges": static_edges    
        }

        # 7. Optionally Process and Pad Angles
        if self.include_angles:
            angles = self._get_angles(traj_window, n_res)
            native_angles_raw = self._get_angles(native_traj, n_res)
            if angles is None or native_angles_raw is None:
                return None
            native_angles = native_angles_raw.squeeze(0)

            torsion_mask = torch.ones((n_res, 4))
            torsion_mask[0, 0:2] = 0.0
            torsion_mask[-1, 2:4] = 0.0

            if T_frames < self.window_size:
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

        # 1. Map keys to spatial dimension index for dynamic padding
        sp_dim_map = {
            "coords": 1, "native_coords": 0, "res_type": 0, "dssp": 0,
            "residue_idx": 0,
        }
        # add angle mappings if the flag is on
        if self.include_angles:
            sp_dim_map.update({"angles": 1, "native_angles": 0, "torsion_mask": 0})

        # 2. Dynamically pad spatial features
        for key in flat_batch[0].keys():
            if key in ["temp", "time_mask", "t", "dataset_idx"]: 
                collated[key] = torch.stack([torch.as_tensor(b[key]) for b in flat_batch])
            elif key == "static_edges":
                continue
            elif key in sp_dim_map:
                padded = []
                for b in flat_batch:
                    L_current = b["res_type"].shape[0]
                    pad_len = max_L - L_current
                    padded.append(self._pad_tensor(b[key], pad_len, spatial_dim=sp_dim_map[key]))
                collated[key] = torch.stack(padded)
        
        # 3. Graph Flattening for Static Edges
        batched_edges = []
        for batch_idx, b in enumerate(flat_batch):
            offset = batch_idx * max_L
            offset_edges = b["static_edges"] + offset 
            batched_edges.append(offset_edges)
            
        collated["static_edges"] = torch.cat(batched_edges, dim=1) 
        
        # 4. Explicitly generate boolean mask
        masks = []
        for b in flat_batch:
            L = b["res_type"].shape[0]
            pad_len = max_L - L
            masks.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
        collated["mask"] = torch.stack(masks).bool()
        
        return collated
    

class FeaturizerSE3Pair(Featurizer):
    '''
    For SE3 Flow Matching
    '''

    def __init__(self, aligner, device='cpu', max_alignment_iters=2, dt_max=340):
        super().__init__(aligner, device=device, max_alignment_iters=max_alignment_iters)
        self.dt_max = dt_max

    def get_time_window(self, total_frames, rng):
        if total_frames <= self.dt_max: return 0, int(total_frames)
        start = rng.integers(0, total_frames - self.dt_max)
        return start, start + self.dt_max

    def process_single(self, xyz, topology):
        '''Extracts features for a single structure'''
        idx_n = topology.select('name N')
        idx_ca = topology.select('name CA')
        idx_c = topology.select('name C')

        R, T = self._construct_rigid_bodies(xyz[idx_n], xyz[idx_ca], xyz[idx_c])
        n_res = R.shape[0]

        t = md.Trajectory(xyz.cpu().numpy().reshape(1, -1, 3), topology)
        
        raw_phi = md.compute_phi(t)[1].flatten()
        raw_psi = md.compute_psi(t)[1].flatten()
        raw_omega = md.compute_omega(t)[1].flatten()

        phi = np.concatenate(([0.0], raw_phi)) 
        psi = np.concatenate((raw_psi, [0.0]))
        omega = np.concatenate((raw_omega, [0.0]))

        torsion_mask = torch.ones((n_res, 6), device=self.device)
        torsion_mask[0, 0:2] = 0.0 
        torsion_mask[-1, 2:4] = 0.0 
        torsion_mask[-1, 4:6] = 0.0 
        
        phi_t = torch.tensor(phi, dtype=torch.float32, device=self.device)
        psi_t = torch.tensor(psi, dtype=torch.float32, device=self.device)
        omega_t = torch.tensor(omega, dtype=torch.float32, device=self.device)

        return R, T, phi_t, psi_t, omega_t, torsion_mask

    def process(self, traj, native_traj, align_target=None, rng=None):

        T_frames = traj.n_frames
        t1 = rng.integers(0, max(1, T_frames - 1))
        t2 = rng.integers(t1 + 1, T_frames) if T_frames > 1 else t1
        
        # Calculate dt
        dt = float(t2 - t1) if t2 > t1 else 1.0 

        xyz_angstrom = torch.tensor(traj.xyz, dtype=torch.float32) * 10.0
        topology = traj.topology

        # ALIGNMENT
        ref_coords = align_target.to(xyz_angstrom.device) if align_target is not None else None
        coords, core_idx = self.aligner.align(xyz_angstrom, ref_coords=ref_coords, core_fraction=0.5, max_iters=self.max_alignment_iters)

        # Process Time Pair
        R_c, T_c, phi_c, psi_c, omega_c, mask_c = self.process_single(coords[t1], topology)
        R_n, T_n, phi_n, psi_n, omega_n, mask_n = self.process_single(coords[t2], topology)

        # Process Native structure — centre on the same stable core used for trajectory alignment
        native_angstrom = torch.tensor(native_traj.xyz, dtype=torch.float32) * 10.0
        native_R, native_coords, native_phi, native_psi, native_omega, _ = self.process_single(native_angstrom[0], topology)
        native_core_mean = native_coords[core_idx].mean(dim=0, keepdim=True)
        native_coords = native_coords - native_core_mean

        # Velocities
        global_disp = T_n - T_c 
        local_disp = torch.matmul(R_c.transpose(-1, -2), global_disp.unsqueeze(-1)).squeeze(-1)
        
        target_v_trans = local_disp / dt
        target_v_rot = self._so3_log_map(R_c, R_n) / dt

        def diff_angle(a, b): 
            return (((b - a + torch.pi) % (2 * torch.pi)) - torch.pi) / dt
        
        v_phi = diff_angle(phi_c, phi_n)
        v_psi = diff_angle(psi_c, psi_n)
        v_omega = diff_angle(omega_c, omega_n)
        
        # Format Angles
        torsion_sincos = torch.stack([
            torch.sin(phi_c), torch.cos(phi_c), 
            torch.sin(psi_c), torch.cos(psi_c),
            torch.sin(omega_c), torch.cos(omega_c)
        ], dim=-1) 

        native_angles = torch.stack([
            torch.sin(native_phi), torch.cos(native_phi), 
            torch.sin(native_psi), torch.cos(native_psi),
            torch.sin(native_omega), torch.cos(native_omega)
        ], dim=-1)

        seq_onehot = self._get_sequence_onehot(topology)#.to(self.device) # keep on cpu

        return {
            "coords": T_c,             
            "native_coords": native_coords,
            "rotation": R_c,
            "native_rot": native_R,       
            "angles": torsion_sincos,  
            "raw_angles_rad": torch.stack([phi_c, psi_c, omega_c], dim=-1),
            "native_angles": native_angles,
            "res_type": seq_onehot, 
            "v_trans": target_v_trans, 
            "v_rot": target_v_rot,     
            "v_ang": torch.stack([v_phi, v_psi, v_omega], dim=-1), 
            "torsion_mask": mask_c
        }
    
    def collate_fn(self, batch):
        flat_batch = [item for sublist in batch if sublist for item in sublist if item]
        if not flat_batch: return None

        max_L = max(b["res_type"].shape[0] for b in flat_batch)
        collated = {}

        for key in flat_batch[0].keys():
            if key in ["temp", "t"]:
                collated[key] = torch.tensor([b[key] for b in flat_batch])
            else:
                padded = []
                for b in flat_batch:
                    L_current = b["res_type"].shape[0]
                    pad_len = max_L - L_current
                    padded.append(self._pad_tensor(b[key], pad_len, spatial_dim=0))
                
                collated[key] = torch.stack(padded)

       # Mask
        masks = []
        for b in flat_batch:
            L = b["res_type"].shape[0]
            pad_len = max_L - L
            masks.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
            
        collated["mask"] = torch.stack(masks).bool()
        
        return collated

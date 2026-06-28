import pytest
import torch
import numpy as np
from pathlib import Path

from dynamode.dataloader.features import Aligner, FeaturizerWindow
from dynamode.dataloader.raw_loader import TrajectoriesDataset
from dynamode.dataloader.zarr_loader import FeaturizerWindowZarr, ZarrTrajectoriesDataset


def _first_existing(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(candidate)
        if p.exists():
            return str(p)
    return None


REPO_ROOT = Path(__file__).resolve().parents[2]
MDCATH_PATH = _first_existing(
    REPO_ROOT / "data" / "mdCATH",
    "/projects/u6hf/hew/mdCATH/data/data",
)
ATLAS_PATH = _first_existing(
    REPO_ROOT / "data" / "ATLAS",
    "/projects/u6hf/hew/ATLAS",
)
ATLAS_ZARR_PATH = _first_existing(
    REPO_ROOT / "data" / "ATLAS" / "atlas_unified.zarr",
    "/projects/u6hf/hew/ATLAS/atlas_unified.zarr",
)
MDCATH_ZARR_PATH = _first_existing(
    REPO_ROOT / "data" / "mdCATH" / "mdcath_unified.zarr",
    "/projects/u6hf/hew/mdCATH/mdcath_unified.zarr",
)


class SoftChecks:
    def __init__(self):
        self.errors = []

    def check(self, cond: bool, msg: str):
        print(("PASS: " if cond else "FAIL: ") + msg)
        if not cond:
            self.errors.append(msg)

    def finish(self):
        if self.errors:
            pytest.fail("Soft checks failed:\n- " + "\n- ".join(self.errors))


def compare_optional_tensor(checks, key, raw_data, zarr_data, atol=1e-4):
    raw_val = raw_data.get(key, None)
    zarr_val = zarr_data.get(key, None)

    if raw_val is None or zarr_val is None:
        checks.check(
            raw_val is None and zarr_val is None,
            f"{key} presence matches: raw={raw_val is not None}, zarr={zarr_val is not None}",
        )
        return

    if not torch.is_tensor(raw_val) or not torch.is_tensor(zarr_val):
        checks.check(raw_val == zarr_val, f"{key} scalar/object matches")
        return

    # Check shapes and RETURN EARLY if they don't match
    shapes_match = raw_val.shape == zarr_val.shape
    checks.check(
        shapes_match,
        f"{key} shape matches: raw={tuple(raw_val.shape)} zarr={tuple(zarr_val.shape)}",
    )
    if not shapes_match:
        return

    if raw_val.dtype == torch.bool or zarr_val.dtype == torch.bool:
        checks.check(torch.equal(raw_val, zarr_val), f"{key} matches exactly")
    else:
        max_diff = torch.abs(raw_val - zarr_val).max().item()
        checks.check(
            torch.allclose(raw_val, zarr_val, atol=atol, rtol=0.0),
            f"{key} allclose within {atol} (max diff={max_diff})",
        )


@pytest.fixture
def align_spy(monkeypatch):
    calls = []
    orig_align = Aligner.align

    def wrapped(self, coords, ref_coords=None, core_fraction=0.5, max_iters=2, apply_coords=None):
        result = orig_align(
            self,
            coords,
            ref_coords=ref_coords,
            core_fraction=core_fraction,
            max_iters=max_iters,
            apply_coords=apply_coords,
        )

        if apply_coords is not None:
            aligned_coords, core_indices, aligned_apply = result
        else:
            aligned_coords, core_indices = result
            aligned_apply = None

        calls.append(
            {
                "self_id": id(self),
                "coords_shape": tuple(coords.shape) if torch.is_tensor(coords) else None,
                "ref_shape": tuple(ref_coords.shape) if torch.is_tensor(ref_coords) else None,
                "core_fraction": core_fraction,
                "max_iters": max_iters,
                "core_indices": core_indices.detach().cpu().tolist() if torch.is_tensor(core_indices) else list(core_indices),
                "aligned_shape": tuple(aligned_coords.shape) if torch.is_tensor(aligned_coords) else None,
                "apply_shape": tuple(apply_coords.shape) if torch.is_tensor(apply_coords) else None,
                "aligned_apply_shape": tuple(aligned_apply.shape) if torch.is_tensor(aligned_apply) else None,
            }
        )
        return result

    monkeypatch.setattr(Aligner, "align", wrapped, raising=True)
    yield calls
    monkeypatch.setattr(Aligner, "align", orig_align, raising=True)



def _batched_kabsch(P, Q):
    """Aligns point cloud P onto Q and returns aligned P."""
    # Center both
    P_mean = P.mean(dim=1, keepdim=True)
    Q_mean = Q.mean(dim=1, keepdim=True)
    P_c = P - P_mean
    Q_c = Q - Q_mean

    # Compute covariance and SVD
    H = torch.bmm(P_c.transpose(1, 2), Q_c)
    U, S, Vh = torch.linalg.svd(H)

    # Optimal rotation
    R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))

    # Correct for reflections
    det = torch.linalg.det(R)
    reflect_mask = det < 0
    if reflect_mask.any():
        Vh_fixed = Vh.clone()
        Vh_fixed[reflect_mask, 2, :] *= -1
        R[reflect_mask] = torch.bmm(Vh_fixed[reflect_mask].transpose(1, 2), U[reflect_mask].transpose(1, 2))

    # Apply rotation and translate to Q's center
    P_aligned = torch.bmm(P_c, R.transpose(1, 2)) + Q_mean
    return P_aligned


@pytest.mark.parametrize("source", ["mdcath", "atlas"])
@pytest.mark.parametrize("coords_type", ["ca", "bb"])
def test_dataloader_comprehensive_validation(source, coords_type):
    """
    Validates that the Zarr pipeline preserves all physical data (geometry, 
    residue types, and native states) relative to the Raw pipeline.
    """
    checks = SoftChecks()

    if MDCATH_PATH is None:
        pytest.skip("mdCATH raw dataset path is not available in this environment")
    if MDCATH_ZARR_PATH is None:
        pytest.skip("mdCATH Zarr dataset path is not available in this environment")
    if source == "atlas":
        if ATLAS_PATH is None:
            pytest.skip("ATLAS raw dataset path is not available in this environment")
        if ATLAS_ZARR_PATH is None:
            pytest.skip("ATLAS Zarr dataset path is not available in this environment")

    # --- 1. Setup DataLoaders ---
    aligner = Aligner()
    featuriser_raw = FeaturizerWindow(
        aligner, window_size=32, include_angles=False, coords_type=coords_type,
    )

    raw_ds = TrajectoriesDataset(
        featuriser=featuriser_raw,
        aligner=aligner,
        mdcath_path=MDCATH_PATH,
        atlas_path=ATLAS_PATH,
        use_zarr=False,
        offline_mode=True,
        use_atlas=True,
        coords_type=coords_type,
        crop=True,
        crop_size=64,
        samples_per_traj=1,
        native_aligned=True,
        atlas_stride=100,
    )

    featuriser_zarr = FeaturizerWindowZarr(
        window_size=32, include_angles=False, coords_type=coords_type,
    )

    zarr_ds = ZarrTrajectoriesDataset(
        featuriser=featuriser_zarr,
        mdcath_zarr_path=MDCATH_ZARR_PATH,
        use_atlas=True,
        atlas_zarr_path=ATLAS_ZARR_PATH,
        coords_type=coords_type,
        crop=True,
        crop_size=64,
        samples_per_traj=1,
        native_aligned=True,
        atlas_stride=100,
    )
    
    # --- 2. Align Index Maps and Extract Batches ---
    test_target = next((item for item in zarr_ds.index_map if item[3] == source), None)
    raw_ds.index_map = [test_target]
    zarr_ds.index_map = [test_target]
    
    torch.manual_seed(42)
    raw_batch = raw_ds[0][0] 
    
    torch.manual_seed(42)
    zarr_batch = zarr_ds[0][0]

    # --- 3. Geometry Validation: Trajectory ---
    # Reshape (T, L, atoms, 3) -> (T, N_atoms, 3)
    T = raw_batch["coords"].shape[0]
    raw_traj_pts = raw_batch["coords"].reshape(T, -1, 3).float()
    zarr_traj_pts = zarr_batch["coords"].reshape(T, -1, 3).float()

    # Align and Compare
    raw_traj_aligned = _batched_kabsch(raw_traj_pts, zarr_traj_pts)
    traj_diff = torch.abs(raw_traj_aligned - zarr_traj_pts).max().item()
    checks.check(
        torch.allclose(raw_traj_aligned, zarr_traj_pts, atol=1e-3),
        f"Trajectory geometry matches (max diff={traj_diff:.5f} Å)"
    )

    # --- 4. Geometry Validation: Native ---
    # Native is (L, atoms, 3). Add batch dim for Kabsch: (1, N_atoms, 3)
    raw_nat_pts = raw_batch["native_coords"].reshape(1, -1, 3).float()
    zarr_nat_pts = zarr_batch["native_coords"].reshape(1, -1, 3).float()

    # Align and Compare
    raw_nat_aligned = _batched_kabsch(raw_nat_pts, zarr_nat_pts)
    nat_diff = torch.abs(raw_nat_aligned[0] - zarr_nat_pts[0]).max().item()
    checks.check(
        torch.allclose(raw_nat_aligned, zarr_nat_pts, atol=1e-4),
        f"Native geometry matches (max diff={nat_diff:.6f} Å)"
    )

    # --- 5. Discrete Metadata Validation ---
    checks.check(
        torch.equal(raw_batch["res_type"], zarr_batch["res_type"]), 
        "Sequence (res_type) matches exactly"
    )

    if "residue_index" in raw_batch:
        checks.check(
            torch.equal(raw_batch["residue_index"], zarr_batch["residue_index"]),
            "Residue indices match exactly"
        )
    
    # Compare masking if available
    if "mask" in raw_batch:
        checks.check(
            torch.equal(raw_batch["mask"], zarr_batch["mask"]),
            "Padding/Missing atom masks match exactly"
        )

    # --- 6. Torsion Angle Validation ---
    # raw_batch['sin_cos'] and zarr_batch['sin_cos'] should match
    if "sin_cos" in raw_batch:
        ang_diff = torch.abs(raw_batch["sin_cos"] - zarr_batch["sin_cos"]).max().item()
        # Use a slightly looser tolerance for angles (1e-3) due to float32 precision
        checks.check(
            torch.allclose(raw_batch["sin_cos"], zarr_batch["sin_cos"], atol=1e-3),
            f"Torsion angles (sin/cos) match (max diff={ang_diff:.5f})"
        )

    # --- 7. Torsion Mask Validation ---
    # Crucial to ensure terminal residues are masked correctly
    if "torsion_mask" in raw_batch:
        checks.check(
            torch.equal(raw_batch["torsion_mask"], zarr_batch["torsion_mask"]),
            "Torsion masks (N/C terminal) match exactly"
        )

    checks.finish()




# @pytest.mark.parametrize("source", ["mdcath", "atlas"])
# @pytest.mark.parametrize("coords_type", ["ca", "bb"])
# def test_dataloader_equivalence(source, coords_type, align_spy, monkeypatch):
#     """
#     Compares raw H5/XTC/PDB features to precomputed Zarr features.
#     Raw path aligns on the fly. Zarr path should already be aligned.
#     """

#     checks = SoftChecks()

#     print(f"\nInitializing Raw Dataset ({source} | {coords_type})...")
#     aligner = Aligner()
#     featuriser_raw = FeaturizerWindow(
#         aligner,
#         window_size=32,
#         include_angles=False,
#         coords_type=coords_type,
#     )

#     raw_ds = TrajectoriesDataset(
#         featuriser=featuriser_raw,
#         aligner=aligner,
#         mdcath_path=MDCATH_PATH,
#         atlas_path=ATLAS_PATH,
#         use_zarr=False,
#         offline_mode=True,
#         use_atlas=True,
#         coords_type=coords_type,
#         crop=True,
#         crop_size=64,
#         samples_per_traj=1,
#         native_aligned=True,
#         atlas_stride=100,
#     )

#     print(f"Initializing Zarr Dataset ({source} | {coords_type})...")
#     featuriser_zarr = FeaturizerWindowZarr(
#         window_size=32,
#         include_angles=False,
#         coords_type=coords_type,
#     )

#     zarr_ds = ZarrTrajectoriesDataset(
#         featuriser=featuriser_zarr,
#         mdcath_zarr_path=MDCATH_ZARR_PATH,
#         use_atlas=True,
#         atlas_zarr_path=ATLAS_ZARR_PATH,
#         coords_type=coords_type,
#         crop=True,
#         crop_size=64,
#         samples_per_traj=1,
#         native_aligned=True,
#         atlas_stride=100,
#     )

#     # Setup target
#     test_target = next((item for item in zarr_ds.index_map if item[3] == source), None)
#     raw_ds.index_map = [test_target]
#     zarr_ds.index_map = [test_target]

#     # --- THE FIX FOR THE TEST ---
#     raw_ds.crop = True
#     zarr_ds.crop = True

#     def fixed_time_window(total_frames, rng):
#         return 0, min(32, total_frames)
#     featuriser_raw.get_time_window = fixed_time_window
#     featuriser_zarr.get_time_window = fixed_time_window

#     # Safely get the global, uncropped native coordinates from Zarr
#     zarr_ds.crop = False
#     full_zarr_batch = zarr_ds[0]
#     global_native = full_zarr_batch[0]["native_coords"].clone()
#     if coords_type == "bb" and global_native.shape[-1] == 12:
#         global_native = global_native.view(global_native.shape[0], 4, 3)
#     zarr_ds.crop = True 

#     # Extract CA-only atoms for the Kabsch reference target
#     if coords_type == "bb":
#         global_native_target = global_native[:, 1, :]
#     else:
#         global_native_target = global_native

#     # ---------------------------------------------------------
#     # SHARED SPATIAL CROP HELPER
#     # ---------------------------------------------------------
#     def apply_manual_spatial_crop(batch, crop_size):
#         for feats in batch:
#             start_res, end_res = 0, crop_size
#             feats["coords"] = feats["coords"][:, start_res:end_res, ...]
#             feats["native_coords"] = feats["native_coords"][start_res:end_res, ...]
#             feats["res_type"] = feats["res_type"][start_res:end_res]
            
#             if "angles" in feats:
#                 feats["angles"] = feats["angles"][:, start_res:end_res, ...]
#                 feats["native_angles"] = feats["native_angles"][start_res:end_res, ...]
#                 feats["torsion_mask"] = feats["torsion_mask"][start_res:end_res, ...]
            
#             if "static_edges" in feats:
#                 edges = feats["static_edges"]
#                 mask = (edges[0] >= start_res) & (edges[0] < end_res) & \
#                         (edges[1] >= start_res) & (edges[1] < end_res)
#                 feats["static_edges"] = edges[:, mask] - start_res
#         return batch

#     # ---------------------------------------------------------
#     # MONKEYPATCH 1: Force Raw Loader to load global, align, then crop
#     # ---------------------------------------------------------
#     orig_process_traj = TrajectoriesDataset._process_trajectory

#     def patched_process_trajectory(self, *args, **kwargs):
#         original_crop = self.crop
#         self.crop = False 
        
#         sub_batch = orig_process_traj(self, *args, **kwargs)
#         self.crop = original_crop
        
#         if self.crop:
#             sub_batch = apply_manual_spatial_crop(sub_batch, self.crop_size)
#         return sub_batch

#     monkeypatch.setattr(TrajectoriesDataset, "_process_trajectory", patched_process_trajectory)

#     # ---------------------------------------------------------
#     # MONKEYPATCH 2: Force Zarr Loader to bypass RNG and crop identically
#     # ---------------------------------------------------------
#     orig_zarr_getitem = ZarrTrajectoriesDataset.__getitem__

#     def patched_zarr_getitem(self, idx):
#         original_crop = self.crop
#         self.crop = False
        
#         # Fetch the full sequence from Zarr bypassing any RNG crops
#         batch = orig_zarr_getitem(self, idx)
#         self.crop = original_crop
        
#         if self.crop:
#             batch = apply_manual_spatial_crop(batch, self.crop_size)
#         return batch

#     monkeypatch.setattr(ZarrTrajectoriesDataset, "__getitem__", patched_zarr_getitem)

#     # ---------------------------------------------------------
#     # MONKEYPATCH 3: Force Featurizer to align to the Global Native CA
#     # ---------------------------------------------------------
#     orig_featurizer_process = featuriser_raw.process

#     def patched_featurizer_process(traj_window, native_traj, align_target=None, rng=None):
#         return orig_featurizer_process(traj_window, native_traj, align_target=global_native_target, rng=rng)

#     monkeypatch.setattr(featuriser_raw, "process", patched_featurizer_process)
#     featuriser_raw.max_alignment_iters = 1

#     print("Dataset settings:")
#     print(f"  raw  -> crop={raw_ds.crop}, crop_size={raw_ds.crop_size}, native_aligned={raw_ds.native_aligned}")
#     print(f"  zarr -> crop={zarr_ds.crop}, crop_size={zarr_ds.crop_size}, native_aligned={zarr_ds.native_aligned}")
#     checks.check(raw_ds.crop == zarr_ds.crop, "Raw and zarr crop flags match")
#     checks.check(raw_ds.crop_size == zarr_ds.crop_size, "Raw and zarr crop_size match")

#     print(f"Fetching idx=0 ({test_target}) from raw loader...")
#     align_start_raw = len(align_spy)
#     raw_batch = raw_ds[0]
#     align_end_raw = len(align_spy)
#     raw_align_calls = align_spy[align_start_raw:align_end_raw]

#     print(f"Fetching idx=0 ({test_target}) from zarr loader...")
#     align_start_zarr = len(align_spy)
#     zarr_batch = zarr_ds[0]
#     align_end_zarr = len(align_spy)
#     zarr_align_calls = align_spy[align_start_zarr:align_end_zarr]

#     checks.check(len(raw_batch) == 1, "Raw loader returned one sample")
#     checks.check(len(zarr_batch) == 1, "Zarr loader returned one sample")
#     if len(raw_batch) != 1 or len(zarr_batch) != 1:
#         checks.finish()

#     raw_data = raw_batch[0]
#     zarr_data = zarr_batch[0]

#     print("Raw batch keys:", sorted(raw_data.keys()))
#     print("Zarr batch keys:", sorted(zarr_data.keys()))

#     checks.check(
#         raw_data["win_pos"] == zarr_data["win_pos"],
#         f"Window position matches: raw={raw_data['win_pos']} zarr={zarr_data['win_pos']}",
#     )

#     print(f"Raw coords shape:  {tuple(raw_data['coords'].shape)}")
#     print(f"Zarr coords shape: {tuple(zarr_data['coords'].shape)}")
#     print(f"Raw native shape:  {tuple(raw_data['native_coords'].shape)}")
#     print(f"Zarr native shape: {tuple(zarr_data['native_coords'].shape)}")

#     checks.check(
#         raw_data["coords"].shape == zarr_data["coords"].shape,
#         f"Trajectory coord shapes match: raw={tuple(raw_data['coords'].shape)} zarr={tuple(zarr_data['coords'].shape)}",
#     )
#     checks.check(
#         raw_data["native_coords"].shape == zarr_data["native_coords"].shape,
#         f"Native coord shapes match: raw={tuple(raw_data['native_coords'].shape)} zarr={tuple(zarr_data['native_coords'].shape)}",
#     )

#     checks.check(torch.equal(raw_data["res_type"], zarr_data["res_type"]), "res_type matches exactly")
#     compare_optional_tensor(checks, "time_mask", raw_data, zarr_data, atol=0.0)
#     compare_optional_tensor(checks, "static_edges", raw_data, zarr_data, atol=0.0)
#     compare_optional_tensor(checks, "temp", raw_data, zarr_data, atol=0.0)

#     print(f"Raw align calls captured:  {len(raw_align_calls)}")
#     print(f"Zarr align calls captured: {len(zarr_align_calls)}")
#     checks.check(len(raw_align_calls) > 0, "Raw pipeline used Aligner.align")
#     checks.check(len(zarr_align_calls) == 0, "Zarr pipeline did not realign during sampling")

#     if raw_align_calls:
#         print("Raw align call[0]:")
#         print(f"  coords_shape={raw_align_calls[0]['coords_shape']}")
#         print(f"  ref_shape={raw_align_calls[0]['ref_shape']}")
#         print(f"  core_fraction={raw_align_calls[0]['core_fraction']}")
#         print(f"  max_iters={raw_align_calls[0]['max_iters']}")
#         print(f"  core_indices(first 20)={raw_align_calls[0]['core_indices'][:20]}")
#         print("Raw core indices:", raw_align_calls[0]["core_indices"])
#     else:
#         print("Raw align call[0]: none captured")

#     print("Raw native_coords mean:", raw_data["native_coords"].mean(dim=0))
#     print("Zarr native_coords mean:", zarr_data["native_coords"].mean(dim=0))
#     print("Raw native_coords first rows:", raw_data["native_coords"][:5])
#     print("Zarr native_coords first rows:", zarr_data["native_coords"][:5])

#     print(f"Crop settings in this test: raw={raw_ds.crop}, zarr={zarr_ds.crop}")
#     checks.check(raw_ds.crop is True and zarr_ds.crop is True, "Crop enabled to test correct spatial chunking")

#     # ---------------------------------------------------------
#     # FINAL FIX: Center coordinates to remove global translation
#     # ---------------------------------------------------------
#     def center_coords(coords, native):
#         # Flatten to (-1, 3) to find the global XYZ centroid, then subtract it
#         centroid = native.reshape(-1, 3).mean(dim=0)
        
#         native_centered = (native.reshape(-1, 3) - centroid).reshape(native.shape)
        
#         # coords shape is (Time, Res, Atoms*3), reshape to (-1, 3) to broadcast subtraction
#         T, L, _ = coords.shape
#         coords_centered = (coords.reshape(T, L, -1, 3) - centroid).reshape(coords.shape)
        
#         return coords_centered, native_centered

#     raw_coords, raw_native = center_coords(raw_data["coords"], raw_data["native_coords"])
#     zarr_coords, zarr_native = center_coords(zarr_data["coords"], zarr_data["native_coords"])

#     tolerance = 1e-4

#     print("Comparing native_coords (centered)...")
#     diff_native = torch.abs(raw_native - zarr_native).max().item()
#     checks.check(
#         torch.allclose(raw_native, zarr_native, atol=tolerance, rtol=0.0),
#         f"native_coords allclose within {tolerance} (max diff={diff_native})",
#     )

#     print("Comparing trajectory coords (centered)...")
#     diff_coords = torch.abs(raw_coords - zarr_coords).max().item()
#     checks.check(
#         torch.allclose(raw_coords, zarr_coords, atol=tolerance, rtol=0.0),
#         f"coords allclose within {tolerance} (max diff={diff_coords})",
#     )

#     checks.check(
#         raw_data["coords"].shape[1] == raw_data["native_coords"].shape[0],
#         "Raw coords and native_coords have matching residue length",
#     )
#     checks.check(
#         zarr_data["coords"].shape[1] == zarr_data["native_coords"].shape[0],
#         "Zarr coords and native_coords have matching residue length",
#     )

#     print(f"Done testing {source} with {coords_type}.")
#     print(f"  raw coords shape:  {tuple(raw_data['coords'].shape)}")
#     print(f"  zarr coords shape: {tuple(zarr_data['coords'].shape)}")

#     checks.finish()

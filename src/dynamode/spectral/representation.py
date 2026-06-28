'''
This module handles the encoding and decoding of input Ca coordinates into whatever specified 
representation is desired for the model. These representations include:
- RAW_COORDS: the raw aligned coordinates, which are the absolute coordinates of the CA atoms in the trajectory.
- DISPLACEMENT: the displacement of the CA atoms from their native positions, which is the difference between the aligned coordinates and the native coordinates.
- UNIT_CHAIN_MEAN: a unit-chain representation that preserves the mean bond lengths of the CA atoms in the trajectory.
- UNIT_CHAIN_NATIVE: a unit-chain representation that preserves the native bond lengths of the CA atoms in the trajectory.
- UNIT_CHAIN_PRED: a unit-chain representation that preserves the predicted bond lengths of the CA atoms in the trajectory

It also handles the spectral transforms and normalization of the spectral coefficients, as well as 
any DC residualization that may be desired. 

Spectral transform machinery from the adapters module is used to convert between time and spectral 
representations, and the conditioned_freq_scale is used for scale factor loading and normalisation.
These operations are unified here under SpectralRepresentationPipeline producing the the input that 
the model directly reasons over.

NOTE the UNIT_CHAIN representation inherently changes the data by normalising bond lengths to 
canonical lengths. RAW_COORDS and DISPLACEMENT are exact encode/decodes.
'''


from __future__ import annotations
from typing import Any
import torch

from dynamode.spectral.conditioned_freq_scale import ConditionedFreqScaleLookup
from dynamode.spectral.adapters import DCT, DFT
from dynamode.spectral.aliases import *
from dynamode.utils import ca_bond_dirs, ca_bond_lengths, chain_from_anchor_dirs_lengths


# Representation config specification parsers

def canonical_representation(name=None, *, displacement=None):
    '''Interface to resolve config representation names, defaults to displacement from native'''
    if name is None:
        return DISPLACEMENT if bool(displacement) else RAW_COORDS
    key = str(name).strip().lower()
    if key not in ALIASES:
        valid = ", ".join(sorted(set(ALIASES.values())))
        raise ValueError(f"Unknown representation={name!r}. Expected one of: {valid}")
    return ALIASES[key]

def _canonical_policy(value, aliases, *, name):
    key = str(value or "auto").strip().lower()
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown {name}={value!r}. Expected one of: {valid}")
    return aliases[key]

def canonical_freq_normalization(value="auto"):
    return _canonical_policy(value, NORMALIZATION_ALIASES, name="freq_normalization")

def canonical_dc_residualization(value="auto"):
    return _canonical_policy(value, DC_ALIASES, name="dc_residualization")

def canonical_aniso_source(value="auto"):
    return _canonical_policy(value, ANISO_ALIASES, name="aniso_source")


class CoordinateRepresentation:
    '''
    Encode/decode coordinate representations used before spectral transforms.

    The unit-chain representations preserve the dataloader's residue axis:
    slot 0 stores the anchor displacement of CA_0 from native CA_0; slots
    1..L-1 store the unit direction of the preceding CA-CA bond. The
    predicted-length variant adds a fourth channel containing a bounded
    residual around native CA-CA length.
    '''

    def __init__(
        self,
        representation=None,
        *,
        displacement=None,
        coord_channels=3,
        length_min=3.5,
        length_max=4.1,
        length_residual_max=0.30,
        eps=1e-8,
    ):
        self.name = canonical_representation(representation, displacement=displacement)
        self.raw_coord_channels = int(coord_channels)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.length_residual_max = float(length_residual_max)
        self.eps = float(eps)

        if self.is_unit_chain and self.raw_coord_channels != 3:
            raise ValueError(
                f"{self.name} requires CA-only coords with 3 channels; got coord_channels={coord_channels}"
            )

    @property
    def is_displacement(self):
        return self.name == DISPLACEMENT

    @property
    def is_unit_chain(self):
        return self.name in {UNIT_CHAIN_MEAN, UNIT_CHAIN_NATIVE, UNIT_CHAIN_PRED}

    @property
    def model_coord_channels(self):
        if self.name == UNIT_CHAIN_PRED:
            return 4
        return self.raw_coord_channels

    def native_lengths(self, native_coords):
        '''Only used by unit chain'''
        return ca_bond_lengths(native_coords, eps=self.eps).clamp(self.length_min, self.length_max)

    def forward(self, coords, native_coords, mask=None, *, return_context=False):
        '''Encode absolute aligned coordinates `(B,T,L,C)` into pre-transform coordinates.'''
        if self.name == RAW_COORDS:
            out = coords
            context = {}
        elif self.name == DISPLACEMENT:
            out = coords - native_coords.unsqueeze(1)
            context = {}
        else:
            if coords.shape[-1] != 3:
                raise ValueError(f"{self.name} expects CA coords, got shape {tuple(coords.shape)}")

            b, t, l, _ = coords.shape
            anchor_disp = coords[:, :, :1, :] - native_coords[:, None, :1, :]
            dirs = ca_bond_dirs(coords, eps=self.eps)
            native_len = self.native_lengths(native_coords)
            mean_len = ca_bond_lengths(coords, eps=self.eps).mean(dim=1).clamp(
                self.length_min, self.length_max
            )
            context = {"native_lengths": native_len, "mean_lengths": mean_len}

            if self.name == UNIT_CHAIN_PRED:
                lengths = ca_bond_lengths(coords, eps=self.eps).clamp(self.length_min, self.length_max)
                residual_unit = ((lengths - native_len.unsqueeze(1)) / self.length_residual_max).clamp(-0.999, 0.999)
                residual = torch.atanh(residual_unit)
                out = coords.new_zeros((b, t, l, 4))
                out[:, :, :1, :3] = anchor_disp
                out[:, :, 1:, :3] = dirs
                out[:, :, 1:, 3] = residual
            else:
                out = coords.new_zeros((b, t, l, 3))
                out[:, :, :1, :] = anchor_disp
                out[:, :, 1:, :] = dirs

        if mask is not None:
            out = out * self.feature_mask(mask, out.shape[-1]).unsqueeze(1)
        if return_context:
            return out, context
        return out

    encode = forward

    def inverse(self, repr_coords, native_coords, mask=None, context=None):
        '''Decode model-coordinate time series `(B,T,L,C_repr)` to absolute coordinates.'''
        if self.name == RAW_COORDS:
            coords = repr_coords[..., : self.raw_coord_channels]
        elif self.name == DISPLACEMENT:
            coords = repr_coords[..., : self.raw_coord_channels] + native_coords.unsqueeze(1)
        else:
            anchor = native_coords[:, None, 0, :] + repr_coords[:, :, 0, :3]
            dirs = repr_coords[:, :, 1:, :3]
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=-1, keepdim=True).clamp_min(self.eps)

            native_len = self.native_lengths(native_coords)
            if self.name == UNIT_CHAIN_MEAN:
                if context is not None and context.get("mean_lengths") is not None:
                    lengths = context["mean_lengths"].to(device=repr_coords.device, dtype=repr_coords.dtype)
                else:
                    # Generation has no target trajectory from which to infer a mean length;
                    # native lengths are the only available per-bond valid fallback.
                    lengths = native_len
                lengths = lengths.unsqueeze(1).expand(-1, repr_coords.shape[1], -1)
            elif self.name == UNIT_CHAIN_NATIVE:
                lengths = native_len.unsqueeze(1).expand(-1, repr_coords.shape[1], -1)
            else:
                raw_residual = repr_coords[:, :, 1:, 3]
                residual = self.length_residual_max * torch.tanh(raw_residual)
                lengths = (native_len.unsqueeze(1) + residual).clamp(self.length_min, self.length_max)

            coords = chain_from_anchor_dirs_lengths(anchor, dirs, lengths)

        if mask is not None:
            coords = coords * mask[:, None, :, None].to(dtype=coords.dtype)
        return coords

    decode = inverse

    def feature_mask(self, mask, channels=None):
        '''Residue mask (protein length)'''
        channels = int(channels or self.model_coord_channels)
        mask_bool = mask.bool()
        if not self.is_unit_chain:
            return mask_bool.unsqueeze(-1).expand(-1, -1, channels).float()

        b, l = mask_bool.shape
        out = mask_bool.new_zeros((b, l, channels), dtype=torch.float32)
        if l == 0:
            return out
        out[:, 0, : min(3, channels)] = mask_bool[:, 0:1].float()
        if l > 1:
            bond_mask = (mask_bool[:, 1:] & mask_bool[:, :-1]).float()
            out[:, 1:, : min(3, channels)] = bond_mask.unsqueeze(-1)
            if channels > 3:
                out[:, 1:, 3] = bond_mask
        return out

    def spectral_mask(self, mask, torsion_mask, top_k, is_dct):
        '''Residue mask (protein length) on spectral volume'''
        rep_mask = self.feature_mask(mask, self.model_coord_channels)
        if torsion_mask is not None:
            feature_mask = torch.cat([rep_mask, torsion_mask.float()], dim=-1)
        else:
            feature_mask = rep_mask

        if is_dct:
            full = feature_mask.unsqueeze(2).expand(-1, -1, top_k, -1)
            return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)
        full = feature_mask.unsqueeze(2).unsqueeze(-1).expand(-1, -1, top_k, -1, 2)
        return full.reshape(feature_mask.shape[0], feature_mask.shape[1], -1)

    def length_barrier_loss(self, repr_coords, mask=None):
        '''Unit chain only'''
        if self.name != UNIT_CHAIN_PRED or repr_coords.shape[-1] < 4 or repr_coords.shape[2] < 2:
            return repr_coords.new_tensor(0.0)
        raw = repr_coords[:, :, 1:, 3]
        excess = (raw.abs() - 1.0).clamp_min(0.0).square()
        if mask is not None:
            bond_mask = (mask[:, 1:] & mask[:, :-1]).to(dtype=excess.dtype)
            excess = excess * bond_mask.unsqueeze(1)
            return excess.sum() / (bond_mask.sum() * repr_coords.shape[1] + 1e-8)
        return excess.mean()


class SpectralRepresentationPipeline:
    '''
    Central coordinate, spectral transform, normalisation, and DC policy:
    - raw aligned coords -> chosen coordinate representation
    - DCT/DFT time <-> spectral transforms
    - optional amplitude normalisation policy for model internals
    - optional conditioned/per-residue DC residualisation
    - the scale tensor used for anisotropic diffusion noise

    The model-facing scale properties are deliberately separated from the
    internal DC lookup payload so experiments can use, for example, no
    amplitude normalisation while still subtracting per-residue DC baselines.
    '''

    def __init__(
        self,
        *,
        coordinate: CoordinateRepresentation | None = None,
        representation=None,
        displacement=None,
        raw_coord_channels=3,
        total_channels=None,
        use_dct=True,
        scale_factors=None,
        conditioned_freq_scale: dict[str, Any] | None = None,
        freq_normalization="auto",
        dc_residualization="auto",
        aniso_source="auto",
        aniso_scale_factors=None,
        scale_multiplier=1.0,
        device="cpu",
        length_min=3.5,
        length_max=4.1,
        length_residual_max=0.30,
    ):
        self.coordinate = coordinate or CoordinateRepresentation(
            representation,
            displacement=displacement,
            coord_channels=raw_coord_channels,
            length_min=length_min,
            length_max=length_max,
            length_residual_max=length_residual_max,
        )
        self.raw_coord_channels = int(raw_coord_channels)
        self.coord_channels = int(self.coordinate.model_coord_channels)
        self.channels = int(total_channels if total_channels is not None else self.coord_channels)
        self.device = device
        self.scale_multiplier = float(scale_multiplier)
        self.transform_engine = DCT if use_dct else DFT
        self.is_dft = self.transform_engine.__name__ == "DFT"
        self.complex_multiplier = 2 if self.is_dft else 1

        self.freq_normalization = canonical_freq_normalization(freq_normalization)
        self.dc_residualization = canonical_dc_residualization(dc_residualization)
        self.aniso_source = canonical_aniso_source(aniso_source)

        self._freq_scale = self._as_scale(scale_factors)
        self._aniso_scale = self._as_scale(aniso_scale_factors)
        self.conditioned_freq_scale = conditioned_freq_scale
        self._lookup = (
            ConditionedFreqScaleLookup(conditioned_freq_scale)
            if conditioned_freq_scale is not None
            else None
        )

        if self.freq_normalization == "auto":
            self.effective_freq_normalization = "conditioned" if self._lookup is not None else (
                "global" if self._freq_scale is not None else "none"
            )
        else:
            self.effective_freq_normalization = self.freq_normalization

        if self.dc_residualization == "auto":
            has_per_residue_dc = (
                isinstance(conditioned_freq_scale, dict)
                and bool(conditioned_freq_scale.get("per_residue_dc_baselines"))
            )
            if has_per_residue_dc:
                self.effective_dc_residualization = "per_residue"
            elif self._lookup is not None:
                self.effective_dc_residualization = "bucket"
            else:
                self.effective_dc_residualization = "none"
        else:
            self.effective_dc_residualization = self.dc_residualization

        if self.aniso_source == "auto":
            self.effective_aniso_source = "freq_scales" if self._freq_scale is not None else "none"
        else:
            self.effective_aniso_source = self.aniso_source

        if self.effective_freq_normalization in {"global", "conditioned"} and self._freq_scale is None:
            raise ValueError(
                f"freq_normalization={self.freq_normalization!r} requires freq scale factors"
            )
        if self.effective_freq_normalization == "conditioned" and self._lookup is None:
            raise ValueError("freq_normalization='conditioned' requires a conditioned freq-scale payload")
        if self.effective_dc_residualization in {"bucket", "per_residue"} and self._lookup is None:
            raise ValueError(
                f"dc_residualization={self.dc_residualization!r} requires a conditioned freq-scale payload"
            )
        if self.effective_aniso_source == "freq_scales" and self._freq_scale is None:
            raise ValueError("aniso_source='freq_scales' requires freq scale factors")
        if (
            self.effective_aniso_source == "artifact"
            and self._aniso_scale is None
            and self._freq_scale is None
        ):
            raise ValueError("aniso_source='artifact' requires aniso_scales_path or loaded freq scale factors")

    @staticmethod
    def _as_scale(scale):
        if scale is None:
            return None
        return torch.as_tensor(scale).detach().float().contiguous()

    @property
    def name(self):
        return self.coordinate.name

    @property
    def model_coord_channels(self):
        return self.coordinate.model_coord_channels

    @property
    def is_displacement(self):
        return self.coordinate.is_displacement

    @property
    def freq_scale(self):
        '''Scale used by this pipeline's own normalise/denormalise methods.'''
        if self.effective_freq_normalization == "none":
            return None
        return self._freq_scale

    @property
    def model_freq_scale(self):
        '''Global scale tensor to hand to models for internal normalisation.'''
        if self.effective_freq_normalization == "none":
            return None
        return self._freq_scale

    @property
    def model_conditioned_freq_scale(self):
        '''Conditioned scale payload to hand to model batch adapters.'''
        if self.effective_freq_normalization != "conditioned":
            return None
        return self.conditioned_freq_scale

    @property
    def aniso_freq_scale(self):
        if self.effective_aniso_source == "none":
            return None
        if self.effective_aniso_source == "artifact":
            return self._aniso_scale if self._aniso_scale is not None else self._freq_scale
        if self.effective_aniso_source == "freq_scales":
            return self._freq_scale
        return None

    @property
    def per_residue_dc_baselines(self):
        if self.effective_dc_residualization != "per_residue" or not isinstance(self.conditioned_freq_scale, dict):
            return None
        return self.conditioned_freq_scale.get("per_residue_dc_baselines")

    @property
    def use_scale_conditioning(self):
        return self.model_conditioned_freq_scale is not None

    @property
    def scale_condition_modes(self):
        if self._lookup is None or not self.use_scale_conditioning:
            return 0
        return self._lookup.scale_condition_modes

    @property
    def scale_condition_channels(self):
        if self._lookup is None or not self.use_scale_conditioning:
            return 0
        return self._lookup.scale_condition_channels

    def encode(self, coords, native_coords, mask=None, *, return_context=False):
        return self.coordinate.forward(coords, native_coords, mask=mask, return_context=return_context)

    forward = encode

    def decode(self, repr_coords, native_coords, mask=None, context=None):
        return self.coordinate.inverse(repr_coords, native_coords, mask=mask, context=context)

    inverse = decode

    def feature_mask(self, mask, channels=None):
        return self.coordinate.feature_mask(mask, channels=channels)

    def spectral_mask(self, mask, torsion_mask, top_k, is_dct):
        return self.coordinate.spectral_mask(mask, torsion_mask, top_k, is_dct)

    def length_barrier_loss(self, repr_coords, mask=None):
        return self.coordinate.length_barrier_loss(repr_coords, mask=mask)

    def time_to_spectral(self, x_time, top_k=None):
        return self.transform_engine.time_to_spectral(x_time, top_k=top_k)

    def spectral_to_time(self, x_flat, n_time_steps, n_channels=None):
        channels = self.channels if n_channels is None else n_channels
        return self.transform_engine.spectral_to_time(x_flat, n_time_steps, n_channels=channels)

    def resolve_scale(self, x, *, freq_scale_override=None, base_scale=None):
        if self.effective_freq_normalization == "none" and base_scale is None and freq_scale_override is None:
            return None
        scale = freq_scale_override
        if scale is None:
            scale = base_scale if base_scale is not None else self.freq_scale
        if scale is None:
            return None
        scale = scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-8)
        if scale.ndim == 1:
            return scale.view(1, 1, -1)
        if scale.ndim == 2:
            return scale.unsqueeze(1)
        raise ValueError(f"freq scale must be 1D or 2D, got {tuple(scale.shape)}")

    def normalise(self, x, *, freq_scale_override=None, base_scale=None):
        scale = self.resolve_scale(x, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        return (x / scale) * self.scale_multiplier

    def denormalise(self, x, *, freq_scale_override=None, base_scale=None):
        scale = self.resolve_scale(x, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        return (x / self.scale_multiplier) * scale

    def denormalise_modes(self, x, *, top_k_freqs, in_channels, freq_scale_override=None, base_scale=None):
        if x.ndim != 4:
            raise ValueError(f"denormalise_modes expected 4D input, got {tuple(x.shape)}")
        flat_view = x.view(x.shape[0], x.shape[1], -1)
        scale = self.resolve_scale(flat_view, freq_scale_override=freq_scale_override, base_scale=base_scale)
        if scale is None:
            return x
        if scale.shape[0] == 1:
            scale = scale.view(1, top_k_freqs, in_channels)[:, : x.shape[2], :]
            return (x / self.scale_multiplier) * scale.view(1, 1, x.shape[2], in_channels)
        scale = scale.squeeze(1).view(x.shape[0], top_k_freqs, in_channels)[:, : x.shape[2], :]
        return (x / self.scale_multiplier) * scale.view(x.shape[0], 1, x.shape[2], in_channels)

    def lookup_scales(self, temp, mask, *, seq_len=None, device=None):
        if self._lookup is None or self.effective_freq_normalization != "conditioned":
            return None
        return self._lookup.lookup_scales(temp, mask, seq_len=seq_len, device=device)

    def lookup_scale_features(self, temp, mask, *, seq_len=None, device=None):
        if self._lookup is None or self.effective_freq_normalization != "conditioned":
            return None
        return self._lookup.lookup_scale_features(temp, mask, seq_len=seq_len, device=device)

    def lookup_model_conditioning(self, temp, mask, *, seq_len=None, device=None):
        return (
            self.lookup_scales(temp, mask, seq_len=seq_len, device=device),
            self.lookup_scale_features(temp, mask, seq_len=seq_len, device=device),
        )

    def lookup_dc_baselines(self, temp, mask, *, coord_channels=None, seq_len=None, device=None):
        if self._lookup is None or self.effective_dc_residualization == "none":
            return None
        return self._lookup.lookup_dc_baselines(
            temp,
            mask,
            coord_channels=coord_channels or self.coord_channels,
            seq_len=seq_len,
            device=device,
        )

    def residualise_dc(self, x, temp, mask, *, coord_channels=None, seq_len=None, per_residue_baseline=None):
        if self._lookup is None or self.effective_dc_residualization == "none":
            return x, None
        if self.effective_dc_residualization == "bucket":
            per_residue_baseline = None
        return self._lookup.residualise_dc(
            x,
            temp,
            mask,
            coord_channels=coord_channels or self.coord_channels,
            seq_len=seq_len,
            per_residue_baseline=per_residue_baseline,
        )

    def restore_dc(self, x, baseline, *, coord_channels=None):
        if self._lookup is None or self.effective_dc_residualization == "none" or baseline is None:
            return x
        return self._lookup.restore_dc(
            x,
            baseline,
            coord_channels=coord_channels or self.coord_channels,
        )

    def forward_transform(self, x_time, mask, top_k=None):
        if mask is not None:
            x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
        x_flat = self.time_to_spectral(x_time, top_k=top_k)
        if mask is not None:
            x_flat = x_flat * mask.unsqueeze(-1)
        return self.normalise(x_flat)

    def inverse_transform(self, x_norm, n_time_steps, mask=None, dynamic_gain=1.0, gain_start_k=None, freq_scale_override=None):
        del dynamic_gain, gain_start_k
        x_spec = self.denormalise(x_norm, freq_scale_override=freq_scale_override)
        if not torch.isfinite(x_spec).all():
            x_spec = torch.nan_to_num(x_spec, nan=0.0, posinf=100.0, neginf=-100.0)
        x_time = self.spectral_to_time(x_spec, n_time_steps, n_channels=self.channels)
        if mask is not None:
            x_time = x_time * mask.unsqueeze(1).unsqueeze(-1)
        result = {"coords": x_time[..., : self.coord_channels]}
        if self.channels > self.coord_channels:
            result["angles"] = x_time[..., self.coord_channels :]
        return result

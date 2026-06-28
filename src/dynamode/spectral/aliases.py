RAW_COORDS = "raw_coords"
DISPLACEMENT = "displacement"
UNIT_CHAIN_MEAN = "unit_chain_mean_lengths"
UNIT_CHAIN_NATIVE = "unit_chain_native_lengths"
UNIT_CHAIN_PRED = "unit_chain_pred_lengths"


ALIASES = {
    "raw": RAW_COORDS,
    "coords": RAW_COORDS,
    "raw_coords": RAW_COORDS,
    "absolute": RAW_COORDS,
    "displacement": DISPLACEMENT,
    "native_displacement": DISPLACEMENT,
    "unit_chain_mean": UNIT_CHAIN_MEAN,
    "unit_chain_mean_lengths": UNIT_CHAIN_MEAN,
    "unit_chain_native": UNIT_CHAIN_NATIVE,
    "unit_chain_native_lengths": UNIT_CHAIN_NATIVE,
    "unit_chain_pred": UNIT_CHAIN_PRED,
    "unit_chain_pred_lengths": UNIT_CHAIN_PRED,
    "unit_chain_residual_lengths": UNIT_CHAIN_PRED,
}


NORMALIZATION_ALIASES = {
    "auto": "auto",
    "none": "none",
    "off": "none",
    "identity": "none",
    "global": "global",
    "freq": "global",
    "freq_scales": "global",
    "conditioned": "conditioned",
    "conditioned_freq_scales": "conditioned",
}

DC_ALIASES = {
    "auto": "auto",
    "none": "none",
    "off": "none",
    "bucket": "bucket",
    "conditioned": "bucket",
    "per_residue": "per_residue",
    "per-residue": "per_residue",
}

ANISO_ALIASES = {
    "auto": "auto",
    "none": "none",
    "off": "none",
    "freq_scales": "freq_scales",
    "model": "freq_scales",
    "normalization": "freq_scales",
    "artifact": "artifact",
}
# GroupPose transformer / criterion package.
# Applying the xtcocotools compatibility shim here ensures it is active
# whenever GroupPoseHead (which imports this package) is loaded.
try:
    import xtcocotools.cocoeval as _xt_cocoeval
    from grouppose._xtcocotools_patch import COCOeval as _PatchedCOCOeval
    _xt_cocoeval.COCOeval = _PatchedCOCOeval
except ImportError:
    pass

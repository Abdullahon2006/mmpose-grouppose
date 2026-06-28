# Compatibility shim: xtcocotools.cocoeval -> pycocotools.cocoeval
# Extended to accept sigmas and use_area args that mmpose passes.
import numpy as np
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCOeval(_COCOeval):
    """Thin wrapper adding sigmas and use_area constructor args."""

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm',
                 sigmas=None, use_area=True):
        super().__init__(cocoGt, cocoDt, iouType)
        if sigmas is not None:
            self.params.kpt_oks_sigmas = np.array(sigmas)
        self.use_area = use_area

    def _computeOks(self, imgId, catId):
        """Override to respect use_area flag."""
        p = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas if hasattr(p, 'kpt_oks_sigmas') \
            else np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72,
                           .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        vars = (sigmas * 2) ** 2
        for j, gt in enumerate(gts):
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros(k1)
                    dx = np.max((z, x0 - xd), axis=0) + \
                        np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + \
                        np.max((z, yd - y1), axis=0)
                if self.use_area:
                    e = (dx ** 2 + dy ** 2) / vars / \
                        (gt['area'] + np.spacing(1)) / 2
                else:
                    s = (bb[2] * bb[3])
                    e = (dx ** 2 + dy ** 2) / vars / (s + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

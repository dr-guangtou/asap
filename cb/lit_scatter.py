# Literature values of the scatter in SM at fixed HM
# Only care about HM > 13
import numpy as np

hm0, hm1, n = 13, 15, 50

lit = {
        # "gu2016": {
        #     "x": np.linspace(hm0, hm1, num=n),
        #     "y": np.array([0.2] * n),
        #     "label": "Gu2016",
        # },
        "kravtsov2018a": {
            "x": np.linspace(hm0, hm1, num=n),
            "y": np.array([0.2] * n),
            "label": r"Kravtsov2018 $M_{\ast}^{cen}$",
        },
        "kravtsov2018b": {
            "x": np.linspace(hm0, hm1, num=n),
            "y": np.array([0.1] * n),
            "label": r"Kravtsov2018 $M_{\ast}^{halo}$",
        },
        # Note that this is at slightly higher Z (0 - 0.6) than others
        # But they say there is no redshift evolution so that is OK
        "lin2012": {
            "x": np.linspace(14, np.log10(2e15), num=n),
            "y": np.array([0.12] * n),
            "label": "Lin2012",
        },
        "leauthaud2012": {
            "x": np.linspace(hm0, hm1, num=n),
            "y": np.array([0.23] * n),
            "label": "Leauthaud2012",
        },
}

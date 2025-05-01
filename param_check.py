import pandas as pd
from .config import PARAMS

df = pd.read_csv(PARAMS)

impr = df["mean_norm_improvement"]

mean   = impr.mean()
std    = impr.std(ddof=0)
_min   = impr.min()
_max   = impr.max()
_range = _max - _min

# relative metrics
coef_var    = std / mean if mean != 0 else float("nan")
range_ratio = _range / mean if mean != 0 else float("nan")

print(f"n runs               : {len(impr)}")
print(f"mean improvement     : {mean:.6f}")
print(f"std. dev.            : {std:.6f}")
print(f"min / max improvement: {_min:.6f} / {_max:.6f}")
print(f"absolute range       : {_range:.6f}")
print(f"coef. of variation   : {coef_var:.2%}")
print(f"range ⁄ mean         : {range_ratio:.2%}")

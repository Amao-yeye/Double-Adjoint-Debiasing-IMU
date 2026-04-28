import pickle
import os

path = 'results/Euroc_baseline_w16/Sequnces-results/MH_02_easy/FullTra_result.p'
with open(path, 'rb') as f:
    data = pickle.load(f)

print("Type:", type(data).__name__)
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    for k, v in data.items():
        t = type(v).__name__
        if hasattr(v, 'shape'):
            print(f"  {k}: {t}, shape={v.shape}")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {t}, value={v}")
        elif isinstance(v, dict):
            subkeys = list(v.keys())[:10]
            print(f"  {k}: {t}, keys={subkeys}")
            for sk, sv in list(v.items())[:3]:
                st = type(sv).__name__
                if hasattr(sv, 'shape'):
                    print(f"    {sk}: {st}, shape={sv.shape}")
                elif isinstance(sv, (int, float)):
                    print(f"    {sk}: {st}, value={sv}")
                else:
                    print(f"    {sk}: {st}, repr={repr(sv)[:80]}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: {t}, len={len(v)}")
        else:
            print(f"  {k}: {t}, repr={repr(v)[:80]}")
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    for i, item in enumerate(data[:5]):
        print(f"  [{i}]: type={type(item).__name__}")
        if isinstance(item, dict):
            print(f"       keys={list(item.keys())}")

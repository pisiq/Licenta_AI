import json, glob, sys

files = sorted(glob.glob('C:/Facultate/Licenta/data/iclr_2017/train/reviews/*.json'))
print(f"Found {len(files)} ICLR train review files")
for fpath in files[:5]:
    with open(fpath, encoding='utf-8') as fp:
        d = json.load(fp)
    reviews = d.get('reviews', [])
    print(f"\n{fpath.split('/')[-1]}: {len(reviews)} reviewers")
    for i, r in enumerate(reviews[:2]):
        print(f"  Reviewer {i}: keys={list(r.keys())}")
        # Print any numeric fields
        for k, v in r.items():
            if k != 'comments' and k != 'IS_META_REVIEW':
                print(f"    {k} = {repr(v)}")


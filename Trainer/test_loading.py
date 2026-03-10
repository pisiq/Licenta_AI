"""Quick test to verify data loading works correctly."""
from data_preprocessing import load_peerread_data, TextPreprocessor, PEERREAD_ALL_CONFERENCES

tp = TextPreprocessor(max_length=5000, min_length=50)
data = load_peerread_data(
    base_data_path='../data',
    text_preprocessor=tp,
    require_pdf=True,
    verbose=True,
    seed=42,
)

by_conf = {}
by_split = {}
for s in data:
    by_conf[s.conference] = by_conf.get(s.conference, 0) + 1
    by_split[s.split] = by_split.get(s.split, 0) + 1

print('\nBy conference:', by_conf)
print('By split:     ', by_split)

rec_valid = sum(1 for s in data if s.score_mask.get('RECOMMENDATION', False))
print(f'\nRECOMMENDATION valid: {rec_valid}/{len(data)}')

# Show sample from each source
for conf_prefix in ['acl_2017', 'conll_2016', 'ICLR_2017', 'ICLR_2018']:
    samples = [s for s in data if s.conference == conf_prefix]
    if samples:
        s = samples[0]
        rec = s.scores.get('RECOMMENDATION')
        print(f'{conf_prefix:15} -> {s.paper_id}  REC={rec:.2f}  split={s.split}  mask={s.score_mask}')


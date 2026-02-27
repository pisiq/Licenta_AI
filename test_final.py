from data_preprocessing import TextPreprocessor, load_peerread_data, PaperReviewDataset, split_data

preprocessor = TextPreprocessor(max_length=10000, min_length=50)
data = load_peerread_data('./data', preprocessor, verbose=True)

print("TOTAL=" + str(len(data)))
s = data[0]
print("paper_id=" + s.paper_id + "  conf=" + s.conference + "  split=" + s.split)
print("body_len=" + str(len(s.paper_text)))
print("review_len=" + str(len(s.review_comments)))
print("combined_len=" + str(len(s.combined_text)))
print("combined_preview=" + s.combined_text[:200])
valid = {k: round(v, 2) for k, v in s.scores.items() if v is not None}
print("scores=" + str(valid))
print("all_8_scored=" + str(all(s.score_mask.values())))

train_data, dev_data, test_data = split_data(data)
print("train=" + str(len(train_data)) + "  dev=" + str(len(dev_data)) + "  test=" + str(len(test_data)))

ds = PaperReviewDataset(train_data, None, print_summary=True)


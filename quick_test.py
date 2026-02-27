"""Quick test of data loading."""
from data_preprocessing import load_all_peerread_data, TextPreprocessor, ReviewAggregator

# Create log file
log = open('test_output.txt', 'w', encoding='utf-8')

def log_print(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

log_print("Creating preprocessors...")
tp = TextPreprocessor()
ra = ReviewAggregator()

log_print("Loading data...")
data = load_all_peerread_data('./data', tp, ra)

log_print(f"\n{'='*80}")
log_print(f"FINAL RESULTS: {len(data)} papers loaded")
log_print(f"{'='*80}")

if data:
    log_print(f"\nFirst paper:")
    log_print(f"  Title: {data[0].title[:80]}...")
    log_print(f"  Scores: {data[0].scores}")

log.close()
log_print("Results written to test_output.txt")



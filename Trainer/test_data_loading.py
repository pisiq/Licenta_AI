"""
Test script to verify PeerRead data loading works correctly.
"""
import os
import json
from data_preprocessing import (
    TextPreprocessor,
    ReviewAggregator,
    load_all_peerread_data
)

def inspect_data_structure():
    """Inspect the actual data structure to help debug."""

    # Check one example from each conference
    conferences = [
        'acl_2017',
        'conll_2016',
        'iclr_2017',
        'arxiv.cs.ai_2007-2017',
        'arxiv.cs.cl_2007-2017',
        'arxiv.cs.lg_2007-2017'
    ]

    print("="*80)
    print("INSPECTING DATA STRUCTURE")
    print("="*80)

    for conf in conferences:
        conf_path = os.path.join('../data', conf)

        if not os.path.exists(conf_path):
            print(f"\nX {conf}: Not found")
            continue

        print(f"\n+ {conf}: Found")

        # Check for train/dev/test splits
        for split in ['train', 'dev', 'test']:
            split_path = os.path.join(conf_path, split)
            if os.path.exists(split_path):
                # Count files
                parsed_pdfs_path = os.path.join(split_path, 'parsed_pdfs')
                reviews_path = os.path.join(split_path, 'reviews')

                if os.path.exists(parsed_pdfs_path):
                    pdf_count = len([f for f in os.listdir(parsed_pdfs_path) if f.endswith('.json')])
                else:
                    pdf_count = 0

                if os.path.exists(reviews_path):
                    review_count = len([f for f in os.listdir(reviews_path) if f.endswith('.json')])
                else:
                    review_count = 0

                print(f"  {split:5s}: {pdf_count} PDFs, {review_count} reviews")

                # Load one example to show structure
                if review_count > 0:
                    review_files = [f for f in os.listdir(reviews_path) if f.endswith('.json')]
                    example_review = os.path.join(reviews_path, review_files[0])

                    with open(example_review, 'r', encoding='utf-8') as f:
                        review_data = json.load(f)

                    print(f"    Example review structure: {list(review_data.keys())}")

                    if 'reviews' in review_data and review_data['reviews']:
                        print(f"    First review keys: {list(review_data['reviews'][0].keys())}")

                    # Find corresponding PDF
                    paper_id = review_files[0].replace('.json', '')
                    pdf_file = os.path.join(parsed_pdfs_path, f'{paper_id}.pdf.json')
                    if not os.path.exists(pdf_file):
                        pdf_file = os.path.join(parsed_pdfs_path, f'{paper_id}.json')

                    if os.path.exists(pdf_file):
                        with open(pdf_file, 'r', encoding='utf-8') as f:
                            pdf_data = json.load(f)
                        print(f"    Example PDF structure: {list(pdf_data.keys())}")

def test_loading():
    """Test the actual data loading function."""

    print("\n" + "="*80)
    print("TESTING DATA LOADING")
    print("="*80)

    # Create preprocessors
    text_preprocessor = TextPreprocessor(
        normalize_whitespace=True,
        remove_references=True,
        max_length=10000,
        min_length=100
    )

    review_aggregator = ReviewAggregator(
        method='mean_round',
        min_val=1,
        max_val=5
    )

    # Load all data
    all_data = load_all_peerread_data(
        base_data_path='../data',
        text_preprocessor=text_preprocessor,
        review_aggregator=review_aggregator
    )

    print(f"\n" + "="*80)
    print(f"RESULTS")
    print("="*80)
    print(f"Total papers loaded: {len(all_data)}")

    if len(all_data) > 0:
        print(f"\nFirst paper example:")
        paper = all_data[0]
        print(f"  Title: {paper.title[:100]}...")
        print(f"  Abstract length: {len(paper.abstract)} chars")
        print(f"  Full text length: {len(paper.full_text)} chars")
        print(f"  Scores: {paper.scores}")
    else:
        print("\n[WARNING] No papers were loaded!")
        print("This suggests an issue with the data structure or file format.")

if __name__ == '__main__':
    # First inspect the structure
    inspect_data_structure()

    # Then test loading
    test_loading()




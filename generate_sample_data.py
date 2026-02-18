"""
Utility script to generate sample data for testing the pipeline.

This creates a synthetic dataset in the expected format.
"""
import json
import random
import os


def generate_sample_paper():
    """Generate a sample paper with random text."""

    titles = [
        "Deep Learning for Natural Language Processing",
        "Advances in Computer Vision Techniques",
        "Novel Approaches to Reinforcement Learning",
        "Transformer Models for Machine Translation",
        "Graph Neural Networks in Molecular Design",
        "Attention Mechanisms in Sequence Modeling",
        "Multi-task Learning in Neural Networks",
        "Convolutional Networks for Image Classification",
        "Recurrent Neural Networks for Time Series",
        "Generative Adversarial Networks Applications"
    ]

    abstracts = [
        "This paper presents a novel approach to solving complex problems using deep learning. "
        "We propose a new architecture that outperforms existing methods on several benchmarks. "
        "Our experiments demonstrate significant improvements in both accuracy and efficiency.",

        "We introduce an innovative method for processing sequential data. "
        "The proposed technique leverages attention mechanisms to capture long-range dependencies. "
        "Experimental results show state-of-the-art performance on multiple datasets.",

        "This work explores the application of neural networks to challenging real-world problems. "
        "We develop a framework that combines multiple learning strategies. "
        "Comprehensive evaluations validate the effectiveness of our approach.",

        "We present a comprehensive study of modern machine learning techniques. "
        "Our analysis reveals important insights into model behavior and performance. "
        "The findings contribute to better understanding of deep learning systems.",

        "This research investigates novel architectures for representation learning. "
        "We demonstrate that our method achieves superior results compared to baselines. "
        "The proposed approach is both efficient and scalable."
    ]

    paper_texts = [
        "Introduction: Machine learning has revolutionized many fields. " * 20 +
        "Methods: We propose a novel architecture based on transformers. " * 30 +
        "Experiments: We evaluate on standard benchmarks. " * 25 +
        "Results: Our method achieves state-of-the-art performance. " * 20 +
        "Conclusion: We have demonstrated the effectiveness of our approach. " * 15,

        "Section 1: Background and motivation for this work. " * 25 +
        "Section 2: Related work in the field. " * 20 +
        "Section 3: Our proposed methodology and architecture. " * 35 +
        "Section 4: Experimental setup and datasets. " * 20 +
        "Section 5: Results and analysis. " * 25 +
        "Section 6: Discussion and future directions. " * 15,

        "Abstract provides overview of contributions. " * 10 +
        "Introduction establishes context and motivation. " * 30 +
        "Background reviews relevant prior work. " * 25 +
        "Method describes our novel approach. " * 40 +
        "Evaluation presents comprehensive experiments. " * 30 +
        "Analysis discusses results and implications. " * 20 +
        "Conclusion summarizes key findings. " * 10
    ]

    return {
        "title": random.choice(titles),
        "abstract": random.choice(abstracts),
        "full_text": random.choice(paper_texts)
    }


def generate_sample_reviews(num_reviews=2):
    """Generate sample review scores."""
    dimensions = [
        "IMPACT",
        "SUBSTANCE",
        "APPROPRIATENESS",
        "MEANINGFUL_COMPARISON",
        "SOUNDNESS_CORRECTNESS",
        "ORIGINALITY",
        "CLARITY",
        "RECOMMENDATION"
    ]

    reviews = []
    for _ in range(num_reviews):
        review = {}
        # Generate scores with some correlation (good papers tend to score well across dimensions)
        base_score = random.choice([2, 3, 4])
        for dim in dimensions:
            # Add some noise around base score
            score = base_score + random.randint(-1, 1)
            score = max(1, min(5, score))  # Clip to 1-5
            review[dim] = score
        reviews.append(review)

    return reviews


def generate_sample_dataset(num_papers=100, output_path="./data/papers_reviews.json"):
    """
    Generate a sample dataset for testing.

    Args:
        num_papers: Number of papers to generate
        output_path: Where to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = []

    for i in range(num_papers):
        paper = generate_sample_paper()
        # Randomly assign 1-3 reviews per paper
        num_reviews = random.randint(1, 3)
        reviews = generate_sample_reviews(num_reviews)

        paper['reviews'] = reviews
        dataset.append(paper)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {num_papers} sample papers")
    print(f"Saved to: {output_path}")
    print(f"\nSample entry:")
    print(json.dumps(dataset[0], indent=2))


if __name__ == "__main__":
    generate_sample_dataset(num_papers=100)


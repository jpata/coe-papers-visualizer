
import json
import argparse
from sentence_transformers import SentenceTransformer, util
import torch

# The groups and their descriptions, extracted from diagram.html
# We use the leaf nodes, as they have the descriptive content.
GROUPS = {
    "t1": "Theoretical particle physics",
    "t2": "Gravitational wave phenomenology",
    "t3": "Gravitational theory",
    "l1": "Experimental particle physics",
    "l2": "Hardware development",
    "r1": "Cosmology & astrophysics",
    "r2": "Astronomy",
    "b1": "CMS Tier 2 Computing (ETAIS)",
    "b2": "Computational methods",
    "b3": "Quantum information & computing",
    "cern": "CERN experiments: LHC, CMS, WLCG",
    "esa": "ESA experiments: LISA",
    "other": "Other experiments: 4MOST, JPAS, PTA"
}

def map_papers_to_groups(papers_file, output_file, threshold=0.3, model_name='all-MiniLM-L12-v2'):
    """
    Maps papers to predefined groups based on semantic similarity of their abstracts.

    Args:
        papers_file (str): Path to the input JSON file with paper abstracts.
        output_file (str): Path to save the output JSON mapping.
        threshold (float): Cosine similarity threshold for a match.
        model_name (str): The sentence-transformer model to use.
    """
    print(f"Loading sentence transformer model: {model_name}...")
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(model_name)

    # Load paper data
    print(f"Loading papers from {papers_file}...")
    with open(papers_file, 'r', encoding='utf-8') as f:
        # The JSON file may contain unescaped control characters, which is invalid JSON.
        # `strict=False` allows the parser to handle these gracefully.
        papers_data = json.loads(f.read(), strict=False)

    # Prepare group data
    group_ids = list(GROUPS.keys())
    group_descriptions = list(GROUPS.values())

    # Generate embeddings for all group descriptions (this is done once)
    print("Generating embeddings for group descriptions...")
    group_embeddings = model.encode(group_descriptions, convert_to_tensor=True)

    # Initialize the output mapping
    # The output format is: { "groupId": ["doi1", "doi2", ...], ... }
    output_mapping = {group_id: [] for group_id in group_ids}

    print(f"Processing {len(papers_data)} papers...")

    # Process each paper
    for i, (doi, abstract) in enumerate(papers_data.items()):
        if not abstract or not isinstance(abstract, str) or len(abstract.strip()) == 0:
            print(f"  - Skipping paper {doi} due to empty or invalid abstract.")
            continue

        # Generate embedding for the paper's abstract
        paper_embedding = model.encode(abstract, convert_to_tensor=True)

        # Compute cosine similarity between the paper's abstract and all group descriptions
        cosine_scores = util.pytorch_cos_sim(paper_embedding, group_embeddings)[0]

        # Find matches above the threshold
        for j, score in enumerate(cosine_scores):
            if score.item() > threshold:
                group_id = group_ids[j]
                output_mapping[group_id].append(doi)
                print(f"  - Match found for {doi}: Group '{GROUPS[group_id]}' (Score: {score.item():.2f})")
        
        if (i + 1) % 10 == 0:
            print(f"  ... processed {i + 1}/{len(papers_data)} papers")


    # Save the resulting mapping to a JSON file
    print(f"Saving mapping to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_mapping, f, indent=4)

    print("Processing complete.")
    print(f"Final mapping saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map paper abstracts to predefined research groups using sentence embeddings.")
    parser.add_argument("papers_file", type=str, help="Path to the input JSON file (e.g., 'papers_abstracts.json').")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file (e.g., 'papers_mapped.json').")
    parser.add_argument("--threshold", type=float, default=0.3, help="Cosine similarity threshold for a match (default: 0.3).")
    parser.add_argument("--model", type=str, default='all-MiniLM-L12-v2', help="The sentence-transformer model to use (default: 'all-MiniLM-L12-v2').")

    args = parser.parse_args()

    map_papers_to_groups(
        papers_file=args.papers_file,
        output_file=args.output_file,
        threshold=args.threshold,
        model_name=args.model
    )

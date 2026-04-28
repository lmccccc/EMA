"""
DNF Predicate Generator: generates arbitrary AND/OR predicate combinations.

Output format (predicate_dnf_*.json):
  predicate[query_idx][term_idx][attr_idx] = values
  - Multiple terms are OR'd together (DNF = disjunctive normal form)
  - Within a term, attributes are AND'd
  - [] means "don't care" for that attribute in that term

DNF spec format (--dnf_spec):
  JSON list of objects, each object = one DNF term.
  Keys = attribute indices (string). Values:
    - float  → numerical selectivity (fraction of N)
    - [int]  → categorical label list (AND within)
  Missing keys → don't care.

Examples:
  # (range sel=0.5 AND label=1) OR (label in [2,3])
  --dnf_spec '[{"0": 0.5, "1": [1]}, {"1": [2, 3]}]'

  # range sel=0.3 OR label=5
  --dnf_spec '[{"0": 0.3}, {"1": [5]}]'

  # single AND term (backward compatible)
  --dnf_spec '[{"0": 0.5, "1": [1]}]'
"""

import json
import os
import numpy as np
import argparse
import ast
from utils import read_multy_attr


def arg_init():
    parser = argparse.ArgumentParser(description="DNF Predicate Generator")
    parser.add_argument("--attr_file", type=str, required=True,
                        help="Attribute file path (JSON)")
    parser.add_argument("--attr_type_list", type=str, required=True,
                        help="List of attribute types, 0=numerical, 1=categorical")
    parser.add_argument("--N", type=int, required=True,
                        help="Number of data points")
    parser.add_argument("--query_size", type=int, required=True,
                        help="Number of queries to generate")
    parser.add_argument("--dnf_spec", type=str, required=True,
                        help='DNF spec as JSON, e.g. \'[{"0":0.5,"1":[1]},{"1":[2,3]}]\'')
    parser.add_argument("--predicate_file", type=str, required=True,
                        help="Output predicate file path")
    parser.add_argument("--gt_file", type=str, default=None,
                        help="If set, also generate brute-force groundtruth (requires dataset)")
    return parser.parse_args()


def generate_numerical_predicates(attr_values, selectivity, query_size, N):
    """Generate random range predicates for a numerical attribute."""
    sorted_index = np.argsort(attr_values)
    query_nb = int(N * selectivity)
    query_nb = max(query_nb, 1)
    starts = np.random.randint(0, max(N - query_nb + 1, 1), size=query_size)
    ends = starts + query_nb - 1
    q_ordered_range = np.column_stack([starts, ends])
    q_idx = sorted_index[q_ordered_range]
    query_attr = attr_values[q_idx]
    return [[int(query_attr[i][0]), int(query_attr[i][1])] for i in range(query_size)]


def build_dnf_predicates(attr_type_list, dnf_spec, attr_data, N, query_size):
    """
    Build DNF predicates for all queries.
    Returns: list[query_size] of list[num_terms] of list[num_attrs] of values
    """
    num_attrs = len(attr_type_list)
    num_terms = len(dnf_spec)

    # Pre-extract per-attribute values
    attr_arrays = []
    for idx in range(num_attrs):
        if attr_type_list[idx] == 0:
            attr_arrays.append(np.array([attr_data[i][idx][0] for i in range(N)]))
        else:
            attr_arrays.append(None)  # categorical, not needed for range gen

    # Generate per-term, per-attribute predicates
    # term_preds[term][attr] = list of per-query values
    term_preds = []
    for t, term_spec in enumerate(dnf_spec):
        attr_preds = []
        for a in range(num_attrs):
            key = str(a)
            if key not in term_spec:
                # don't care: empty list for all queries
                attr_preds.append([[] for _ in range(query_size)])
            elif attr_type_list[a] == 0:
                # numerical: selectivity → random ranges
                sel = float(term_spec[key])
                ranges = generate_numerical_predicates(attr_arrays[a], sel, query_size, N)
                attr_preds.append(ranges)
            else:
                # categorical: fixed label list for all queries
                labels = term_spec[key]
                if isinstance(labels, int):
                    labels = [labels]
                attr_preds.append([labels for _ in range(query_size)])
        term_preds.append(attr_preds)

    # Assemble: predicate[query][term][attr]
    predicates = []
    for q in range(query_size):
        q_terms = []
        for t in range(num_terms):
            term = [term_preds[t][a][q] for a in range(num_attrs)]
            q_terms.append(term)
        predicates.append(q_terms)

    return predicates


def compute_selectivity(attr_type_list, attr_data, predicates, N):
    """Estimate actual selectivity by brute-force counting."""
    query_size = len(predicates)
    sample_size = min(query_size, 20)
    sample_idx = np.random.choice(query_size, sample_size, replace=False)

    total_match = 0
    for qi in sample_idx:
        q_pred = predicates[qi]
        match_count = 0
        for i in range(N):
            # OR over terms
            for term in q_pred:
                # AND over attrs
                ok = True
                for a, attr_type in enumerate(attr_type_list):
                    vals = term[a]
                    if not vals:
                        continue
                    if attr_type == 0:
                        v = attr_data[i][a][0]
                        if not (vals[0] <= v <= vals[1]):
                            ok = False
                            break
                    else:
                        item_labels = attr_data[i][a]
                        if not all(l in item_labels for l in vals):
                            ok = False
                            break
                if ok:
                    match_count += 1
                    break
        total_match += match_count

    sel = total_match / (sample_size * N)
    return sel


if __name__ == "__main__":
    args = arg_init()

    attr_type_list = ast.literal_eval(args.attr_type_list)
    dnf_spec = json.loads(args.dnf_spec)
    N = args.N

    print(f"Attribute types: {attr_type_list}")
    print(f"DNF spec ({len(dnf_spec)} terms): {dnf_spec}")
    print(f"N={N}, query_size={args.query_size}")

    attr_data = read_multy_attr(args.attr_file)
    assert len(attr_data) == N

    predicates = build_dnf_predicates(
        attr_type_list, dnf_spec, attr_data, N, args.query_size)

    # Show samples
    print(f"\nSample predicates (first 3 queries):")
    for q in range(min(3, args.query_size)):
        print(f"  query {q}: {predicates[q]}")

    # Estimate selectivity
    print(f"\nEstimating selectivity (sampling 20 queries)...")
    sel = compute_selectivity(attr_type_list, attr_data, predicates, N)
    print(f"  Estimated selectivity: {sel:.4f} ({sel*100:.2f}%)")

    # Save
    with open(args.predicate_file, 'w') as f:
        json.dump(predicates, f)
    print(f"\nDNF predicates saved to {args.predicate_file}")
    print(f"Format: predicate[query][term][attr] = values")

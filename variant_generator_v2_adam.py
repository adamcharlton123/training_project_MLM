#!/usr/bin/env python3

# library modules used in this script:
# csv     -> read mutation data from a CSV file
# random  -> randomly choose mutations and mutation counts
import csv
import random

# ---------------------------------------------------------
# User-configurable parameters
# ---------------------------------------------------------
# Input FASTA sequence. This should contain the reference
# (wild-type) sequence that all variants will be derived from.
FASTA_PATH = "full_seq_HBA_mNeon.fa"

# CSV file containing all possible mutations and their scores.
CSV_PATH = "mut_syn_adam.csv"

# Output file containing generated variants.
OUTPUT_PATH = "variants_syn.tsv"

# Log file used to record summary statistics.
LOG_PATH = "variants_syn.log"

# Total number of variant sequences to generate.
N_VARIANTS = 25000

# Variants are grouped into batches of this size.
# Each batch receives a group mean score.
GROUP_SIZE = 100

# ---------------------------------------------------------
# Mutation count distribution parameters
# ---------------------------------------------------------
# The number of mutations applied to each variant
# follows a normal (Gaussian) distribution.
#
# Example:
# mean = 10 and std = 3 means most variants will have
# roughly 7-13 mutations, with occasional larger or
# smaller values.
MUT_MEAN = 10       # average mutations per variant
MUT_STD = 3         # spread around the mean
MUT_MIN = 1         # prevent variants with zero mutations

# ---------------------------------------------------------
# Utility: safe float parser for synergy values
# ---------------------------------------------------------
# Reads a value that should represent a floating-point number.
#
# Returns:
#   float(value) if conversion succeeds
#   default value otherwise
#
# Useful when CSV fields contain empty strings, "NA",
# "None", or other invalid values.
def parse_float(value, default=0.0):
    if value is None:
        return default

    value = value.strip()

    if value.lower() in ("na", "", "none"):
        return default

    try:
        return float(value)
    except ValueError:
        return default

# ---------------------------------------------------------
# Load original sequence from FASTA
# ---------------------------------------------------------
# FASTA files contain sequence data in the form:
#
# >sequence_name
# ACTGACTG...
#
# Header lines beginning with ">" are ignored.
# All sequence lines are concatenated into a single string.
def load_fasta(path):
    seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and FASTA headers
            if not line or line.startswith(">"):
                continue

            seq.append(line)

    return "".join(seq)

# Load reference sequence once at startup.
ORIGINAL_SEQ = load_fasta(FASTA_PATH)

# ---------------------------------------------------------
# Load mutation table (convert pos 1-based → 0-based)
# Includes synergy fields
# ---------------------------------------------------------
#
# Expected CSV columns:
# pos, from, to, cost, pair, synergy
#
# Example:
# 25,A,G,0.8,120,1.2
#
# Position numbering in biological files is usually 1-based,
# but Python strings use 0-based indexing.
# Therefore positions are converted by subtracting 1.
#
mutations = []

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        mutations.append({
            "pos": int(row["pos"]) - 1,  # convert to Python index
            "from": row["from"],         # original residue/base
            "to": row["to"],             # mutated residue/base
            "cost": float(row["cost"]),  # additive score effect

            # Optional position that interacts with this mutation
            "pair": row.get("pair"),

            # Optional synergy bonus applied when both paired
            # mutations are present in the same variant
            "synergy": parse_float(
                row.get("synergy"), 0.0
            ),
        })

# ---------------------------------------------------------
# Mutation count distribution
# ---------------------------------------------------------
# Randomly determine the number of mutations that will be
# added to a variant sequence.
#
# Uses a Gaussian distribution centred on MUT_MEAN.
#
# The value is forced to be at least MUT_MIN.
def choose_num_mutations():
    k = int(random.gauss(MUT_MEAN, MUT_STD))
    return max(MUT_MIN, k)

# ---------------------------------------------------------
# Generate variants
# ---------------------------------------------------------
# For each variant:
#
# 1. Copy the reference sequence.
# 2. Choose k mutations.
# 3. Apply those mutations.
# 4. Calculate score contributions.
# 5. Apply any synergy bonuses.
# 6. Store results for batch output.
#
total_mutations = 0
total_synergy_events = 0

with open(OUTPUT_PATH, "w") as out:

    # Write output header row
    out.write(
        "id\tsequence\tscore\tn_mutations\t"
        "positions\tsubstitutions\t"
        "group_id\tgroup_mean_score\n"
    )

    # Temporary storage for one batch/group
    group_buffer = []
    group_scores = []

    current_group_id = 1

    # Generate all requested variants
    for vid in range(1, N_VARIANTS + 1):

        # Convert string to list because strings are immutable
        # in Python. A list allows mutation at specific positions.
        seq_list = list(ORIGINAL_SEQ)

        # Running score for this variant
        score = 0.0

        # Determine how many mutations to apply
        k = choose_num_mutations()
        total_mutations += k

        # Randomly select k unique mutations
        chosen = random.sample(mutations, k)

        # Dictionary for fast lookup by position
        chosen_by_pos = {m["pos"]: m for m in chosen}

        # Metadata recorded in output file
        positions_1based = []
        substitutions = []

        # -------------------------------------------------
        # Apply individual mutation effects
        # -------------------------------------------------
        for mut in chosen:

            pos0 = mut["pos"]

            # Modify sequence
            seq_list[pos0] = mut["to"]

            # Add mutation's score contribution
            score += mut["cost"]

            # Store human-readable mutation information
            pos1 = pos0 + 1

            positions_1based.append(str(pos1))

            substitutions.append(
                f"{pos1}:{mut['from']}>{mut['to']}"
            )

        # -------------------------------------------------
        # Apply synergy bonuses
        # -------------------------------------------------
        #
        # If mutation A specifies a paired position B, and
        # both A and B are present, an additional synergy
        # score is added.
        #
        # The condition:
        #     mut["pos"] < pair_pos
        #
        # prevents counting the same mutation pair twice.
        #
        for mut in chosen:

            if mut["pair"]:

                try:
                    pair_pos = int(mut["pair"]) - 1
                except:
                    continue

                if (
                    pair_pos in chosen_by_pos
                    and mut["pos"] < pair_pos
                ):
                    if score >= 0: 
                        score += mut["synergy"]
                    else:
                        score -= mut["synergy"]

                    total_synergy_events += 1

        # Convert sequence list back to string
        variant_seq = "".join(seq_list)

        # Store result until the current group is complete
        group_scores.append(score)

        group_buffer.append([
            vid,
            variant_seq,
            score,
            k,
            ",".join(positions_1based),
            ",".join(substitutions),
        ])

        # -------------------------------------------------
        # When a group is full:
        #   1. Calculate group mean score
        #   2. Write all rows in the group
        #   3. Clear temporary storage
        # -------------------------------------------------
        if vid % GROUP_SIZE == 0:

            group_mean = (
                sum(group_scores) / len(group_scores)
            )

            for row in group_buffer:
                out.write(
                    f"{row[0]}\t{row[1]}\t{row[2]}\t"
                    f"{row[3]}\t{row[4]}\t{row[5]}\t"
                    f"{current_group_id}\t"
                    f"{group_mean}\n"
                )

            # Reset buffers for next group
            group_buffer = []
            group_scores = []

            current_group_id += 1

# ---------------------------------------------------------
# Write summary log file
# ---------------------------------------------------------
# Records high-level statistics about the generation run.
#
# Total variants:
#     Number of sequences generated.
#
# Total mutations applied:
#     Sum of all mutations across all variants.
#
# Total synergy events:
#     Number of paired interactions detected.
#
with open(LOG_PATH, "w") as log:
    log.write(f"Total variants: {N_VARIANTS}\n")
    log.write(
        f"Total mutations applied: {total_mutations}\n"
    )
    log.write(
        f"Total synergy events: "
        f"{total_synergy_events}\n"
    )

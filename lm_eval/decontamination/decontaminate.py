import time
import random
import pickle
import json
import glob
import os
import collections

from .janitor import Janitor, word_ngrams
from .archiver import ZStdTextReader


# Was used for testing the evaluator decoupled from the full logic below
def get_train_overlap_stub(docs, ngrams_path, ngrams_n_size):
    simulated_overlap = 0.1
    contaminated = int(len(docs) * simulated_overlap)
    return random.sample(range(len(docs)), contaminated)


# Returns a dictionary containing all overlapping documents in each
# task. In the standard use case, an overlap occurs when any of the 13-grams
# found in the task document exist in the training set documents.
#
# To generate 13-grams for the pile see scripts/clean_training_data. The final output of these
# scripts are an info.json file containing the n_gram_size (13) and a bunch of "ngrams_{x}.bkt.txt.sorted.zst"
# files. These should exist in the "ngrams_path" provided to this function.

# Algorithm:
# 1. Build lookups for each dataset {ngram: list(document_ids)}
# 2. Merge into an overall lookup {ngram: [(task_name, task_set, doc_ids),]}
# 3. Full scan the 13-grams from the training set against the merged lookup,
#    saving matches in the "duplicates" dictionary {(task_name, task_set): set(doc_ids)}
# 4. Strip the task_set from the dictionary keys and return
#
# We cache the task+set lookups as well as the overlaps.
def get_train_overlap(docs_by_task_set, ngrams_path, limit):
    # return get_train_overlap_stub(docs, ngrams_path, ngrams_n_size)

    info_dict_path = os.path.join(ngrams_path, "info.json")
    info_dict = json.load(open(info_dict_path, "r"))
    ngrams_n_size = info_dict["ngram_size"]

    janitor = Janitor()

    # Build lookup for each dataset first in case we use different task combinations later
    print("Building Lookups...")
    start = time.perf_counter()

    def get_overlaps_dump_path(task_name, task_set, ngrams_n_size, limit):
        return f"data/{task_name}/{task_set}_{ngrams_n_size}grams_limit{limit}.overlaps"

    lookups = {}
    duplicates = {}  # (task_name, task_set): set(doc_ids)}
    sets_to_decontaminate = len(docs_by_task_set.keys())

    for (task_name, task_set), docs in docs_by_task_set.items():
        if not os.path.exists(f"data/{task_name}"):
            os.mkdir(f"data/{task_name}")

        # Check if we've decontaminated this combination before
        overlaps_dump_path = get_overlaps_dump_path(
            task_name, task_set, ngrams_n_size, limit
        )
        if os.path.exists(overlaps_dump_path):
            duplicates[(task_name, task_set)] = pickle.load(
                open(overlaps_dump_path, "rb")
            )
            sets_to_decontaminate -= 1
            continue
        else:
            duplicates[(task_name, task_set)] = set()

        # Build/load the task lookup {ngram: set(documents)}.
        task_set_lookup_path = (
            f"data/{task_name}/{task_set}_{ngrams_n_size}grams_limit{limit}.lookup"
        )
        if os.path.exists(task_set_lookup_path):
            print(f"{task_set_lookup_path} available, loading...")
            lookups[(task_name, task_set)] = pickle.load(
                open(task_set_lookup_path, "rb")
            )
        else:
            print(f"{task_set_lookup_path} not available, building...")
            lookup = collections.defaultdict(set)

            for doc_id, document in enumerate(docs):
                ngrams = word_ngrams(janitor.normalize_string(document), ngrams_n_size)
                for ngram in ngrams:
                    lookup[ngram].add(doc_id)

            pickle.dump(lookup, open(task_set_lookup_path, "wb"))
            lookups[(task_name, task_set)] = lookup

    elapsed = time.perf_counter() - start
    print(f"Building lookups took {elapsed:0.5f} seconds.")

    matched_ngrams = []

    if sets_to_decontaminate > 0:
        print("Merging lookups...")
        start = time.perf_counter()
        merged_lookup = collections.defaultdict(list)
        for (task_name, task_set), lookup in lookups.items():
            for ngram, doc_ids in lookup.items():
                merged_lookup[ngram].append((task_name, task_set, doc_ids))

        elapsed = time.perf_counter() - start
        print(f"Merging lookups took {elapsed:0.5f} seconds.")

        print(f"{ngrams_n_size} grams files found in {ngrams_path}:")
        files = glob.glob(os.path.join(ngrams_path, f"*.sorted.zst"))
        print(files)

        for file in files:
            start = time.perf_counter()
            print(f"Scanning {file}")
            reader = ZStdTextReader(file)
            total_ngrams = 0
            unique_ngrams = 0
            matching_unique = 0
            non_matching_unique = 0

            current_ngram = ""
            for line in reader.read_tqdm():  # Scan training set ngrams file
                total_ngrams += 1
                [ngram, document_id] = line.rsplit(" ", 1)
                if (
                    ngram != current_ngram
                ):  # Only need to match the ngram once in training set
                    unique_ngrams += 1
                    current_ngram = ngram
                    if ngram in merged_lookup:
                        matched_ngrams.append(ngram)  # For logging
                        matching_unique += 1
                        for task_name, task_set, doc_ids in merged_lookup[ngram]:
                            task_doc_set = duplicates[(task_name, task_set)]
                            for (
                                doc_id
                            ) in (
                                doc_ids
                            ):  # Record contamination across all relevant task/set combos
                                task_doc_set.add(doc_id)
                        del merged_lookup[ngram]  # No point matching again
                    else:
                        non_matching_unique += 1

            print(f"Total Ngrams: {total_ngrams}")
            print(f"Unique Ngrams: {unique_ngrams}")
            print(f"Unique Matching: {matching_unique}")
            print(f"Unique Non Matching: {non_matching_unique}")
            print("Matched ngrams:")
            for ngram in matched_ngrams:
                print(ngram)

            elapsed = time.perf_counter() - start
            print(f"Read took {elapsed:0.5f} seconds.")
            print(f"Speed: {(os.path.getsize(file)/1000000.0)/elapsed}MB/second")

        print(duplicates)

        # Dump overlaps separately
        for (task_name, task_set), doc_ids in duplicates.items():
            overlaps_dump_path = get_overlaps_dump_path(
                task_name, task_set, ngrams_n_size, limit
            )
            pickle.dump(doc_ids, open(overlaps_dump_path, "wb"))

    # Strip task set and return
    return {task_name: doc_ids for (task_name, task_set), doc_ids in duplicates.items()}

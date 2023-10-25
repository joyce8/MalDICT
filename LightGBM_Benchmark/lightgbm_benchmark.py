import os
import json
import ember
import argparse
import multiprocessing
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prfs


def get_vec(jsonl):
    """Vectorize a JSON line with EMBER raw features.

    Returns:
    (md5, vector)
    """
    ember_meta = json.loads(jsonl)
    md5 = ember_meta["md5"]
    extractor = ember.PEFeatureExtractor()
    vec = extractor.process_raw_features(ember_meta)
    return md5, vec


def get_tags(tag_path):
    """Make a dict mapping MD5s to ClarAVy tags."""
    md5_tags = {}
    with open(tag_path, "r") as f:
        for jsonl in f:
            entry = json.loads(jsonl.strip())
            md5 = entry["md5"]
            tags = [rank[0] for rank in entry["ranking"]]
            md5_tags[md5] = tags
    return md5_tags


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ember_meta_dir", help="path to directory with raw " +
                        "EMBER metadata .jsonl files (train and test)")
    parser.add_argument("maldict_train_file", help="path to MalDICT .jsonl " +
                        "file with train hashes and tags")
    parser.add_argument("maldict_test_file", help="path to MalDICT .jsonl " +
                        "file with test hashes and tags")
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    # Validate ember_meta_path
    train_path = os.path.join(args.ember_meta_dir, "train_features.jsonl")
    test_path = os.path.join(args.ember_meta_dir, "test_features.jsonl")

    # Read metadata
    with open(train_path, "r") as f:
        train_meta = [line.strip() for line in f]
    with open(test_path, "r") as f:
        test_meta = [line.strip() for line in f]

    # Read train/test tags
    train_md5_tags = get_tags(args.maldict_train_file)
    test_md5_tags = get_tags(args.maldict_test_file)

    # Convert tags to numeric labels
    all_tags = set()
    for tags in train_md5_tags.values():
        all_tags.update(tags)
    for tags in test_md5_tags.values():
        all_tags.update(tags)
    sorted_tags = sorted(all_tags)
    tag_labels = {tag: i for i, tag in enumerate(sorted_tags)}

    # Vectorize EMBER metadata
    pool = multiprocessing.Pool(args.num_processes)
    train_md5_vecs = list(pool.imap(get_vec, train_meta))
    test_md5_vecs = list(pool.imap(get_vec, test_meta))

    # Get sizes of train and test set
    num_train = len(train_md5_vecs)
    num_test = len(test_md5_vecs)
    vec_dim = len(train_md5_vecs[0][1])
    num_labels = len(sorted_tags)

    # Get X and y for train set
    X_train = np.zeros((num_train, vec_dim), dtype=np.float)
    y_train = np.zeros((num_train, num_labels))
    for i, (md5, vec) in enumerate(train_md5_vecs):
        labels = [tag_labels[tag] for tag in train_md5_tags[md5]]
        X_train[i,] = vec
        for j in labels:
            y_train[i,j] = 1

    # Get X and y for test set
    X_test = np.zeros((num_test, vec_dim), dtype=np.float)
    y_test = np.zeros((num_test, num_labels), dtype=np.float)
    for i, (md5, vec) in enumerate(test_md5_vecs):
        labels = [tag_labels[tag] for tag in test_md5_tags[md5]]
        X_test[i,] = vec
        for j in labels:
            y_test[i,j] = 1

    # Load LightGBM config
    with open("lightgbm_config.json", "rb") as f:
        params = json.load(f)
    params.update({"verbose": -1})
    params.update({"num_iterations": 100})

    # Train OvR classifier on each tag
    y_pred = np.zeros((num_test, num_labels))
    for j, tag in enumerate(sorted_tags):

        print("Training classifiers on tag: {}".format(tag))

        # Get train and test sets for fold
        y_train_tag = y_train[:, j]
        y_test_tag = y_test[:, j]

        # Train LightGBM classifier
        train_dataset = lgb.Dataset(X_train, y_train_tag)
        test_dataset = lgb.Dataset(X_test, y_test_tag)
        clf = lgb.train(params, train_dataset)

        # Get predictions and compute accuracy
        predictions = clf.predict(X_test)
        y_pred[:,j] = predictions

    # Get AUC score over all tags
    micro_auc = roc_auc_score(y_test, y_pred, average="micro",
                              multi_class="ovr")
    weighted_auc = roc_auc_score(y_test, y_pred, average="weighted",
                                 multi_class="ovr")

    # Get Precision, Recall, and F1 (assume > 0.5)
    y_pred = (y_pred > 0.5)
    p_micro, r_micro, f1_micro, _ = prfs(y_test, y_pred, average="micro")
    p_avg, r_avg, f1_avg, _ = prfs(y_test, y_pred, average="weighted")

    # Print results
    print("Precision\t{} (micro)\t{} (weighted)".format(p_micro, p_avg))
    print("Recall\t{} (micro)\t{} (weighted)".format(r_micro, r_avg))
    print("F1-Score\t{} (micro)\t{} (weighted)".format(f1_micro, f1_avg))
    print("ROC AUC\t{} (micro)\t{} (weighted)".format(micro_auc, weighted_auc))

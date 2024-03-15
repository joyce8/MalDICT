import os
import json
import argparse
import multiprocessing
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from MalConv import MalConv
import torch.nn.functional as F


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
    parser.add_argument("train_dir", help="Path to directory with files " +
                        "to train on. Directory is traversed recursively.")
    parser.add_argument("test_dir", help="Path to directory with files " +
                        "to test on. Directory is traversed recursively.")
    parser.add_argument("maldict_train_file", help="Path to MalDICT .jsonl " +
                        "file with train hashes and tags")
    parser.add_argument("maldict_test_file", help="Path to MalDICT .jsonl " +
                        "file with test hashes and tags")
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    # Default hyperparameters for MalConv2
    NON_NEG = False
    EMBD_SIZE = 8
    FILTER_SIZE = 512
    FILTER_STRIDE = 512
    NUM_CHANNELS = 128
    EPOCHS = 1
    MAX_FILE_LEN = 16000000
    BATCH_SIZE = 128
    RANDOM_STATE = 42

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
    num_labels = len(sorted_tags)

    # Get train and test datasets of malicious files
    train_dataset = BinaryDataset(args.train_dir, train_md5_tags, num_labels, tag_labels,
                                  max_len=MAX_FILE_LEN)
    test_dataset = BinaryDataset(args.test_dir, test_md5_tags, num_labels, tag_labels,
                                 max_len=MAX_FILE_LEN)

    # Get train and test loaders
    train_sampler = RandomChunkSampler(train_dataset, BATCH_SIZE)
    test_sampler = RandomChunkSampler(test_dataset, BATCH_SIZE)
    loader_threads = max(multiprocessing.cpu_count()-4,
                         multiprocessing.cpu_count()//2+1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              num_workers=loader_threads,
                              collate_fn=pad_collate_func,
                              sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             num_workers=loader_threads,
                             collate_fn=pad_collate_func, sampler=test_sampler)

    # Initialize MalConv2 classifier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MalConv(out_size=num_labels, channels=NUM_CHANNELS,
                    window_size=FILTER_SIZE, stride=FILTER_STRIDE,
                    embd_size=EMBD_SIZE).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    model.train()

    # Train classifier
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Test classifier
    y_test = []
    y_pred = []
    total = 0
    model = model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _, _ = model(inputs)
            outputs = F.sigmoid(outputs)
            B = inputs.shape[0]
            total += B
            y_test += labels.detach().cpu().numpy().tolist()
            y_pred += outputs.detach().cpu().numpy().tolist()
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

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

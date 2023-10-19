# MalDICT

MalDICT is a collection of four datasets, each supporting a different malware classification task. All of the malware is originally from the [VirusShare](https://virusshare.com/) dataset chunks 0 - 465. The labels for these files were obtained using [ClarAVy](https://github.com/NeuromorphicComputationResearchProgram/ClarAVy/tree/master), which uses antivirus scan results for tagging malware.

The MalDICT datasets can be used for benchmarking performance on classifying malware by its behaviors, file properties, vulnerability exploitation, and packers.


## Downloading the Datasets

Hashes and tags for all of the malware in MalDICT are provided in .jsonl files within the ```maldict_tags/``` directory. GIT-LFS is required for downloading these files due to their size. On Debian-based systems, GIT-LFS can be installed using:

```
sudo apt-get install git-lfs
```

After installing GIT-LFS you can download the hashes and tags by cloning this repository:

```
git lfs clone https://github.com/joyce8/MalDICT/
```

All of the malware in MalDICT is part of the VirusShare corpus (chunks 0 - 465). These files can be accessed via [VirusShare](https://virusshare.com/login) or [vx-underground](https://www.vx-underground.org/#E:/root/Samples/Virusshare%20Collection/Downloadable%20Releases).


#### EMBER Raw Metadata

We extracted EMBER metadata from all of the PE files in MalDICT-Behavior, MalDICT-Platform, and MalDICT-Packer. These metadata files can be downloaded via torrent:

```
magnet:?xt=urn:btih:180f1792c392470285cba9b55b4a627f8a2fef58&dn=EMBER%5Fmeta&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80&tr=udp%3A%2F%2Fopen.demonii.com%3A1337&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.leechers-paradise.org%3A6969
```


## Dataset Contents

#### MalDICT-Behavior

MalDICT-Behavior is a dataset of malware tagged according to its category or behavior (e.g. ransomware, downloader, autorun). It 4,317,241 malicious files and 75 separate category/behavior tags. A file may have multiple tags if it exhibits more than one type of behavior.

A default train/test split for MalDICT-Behavior is provided in ```MalDICT_Tags/maldict_category_train.jsonl``` and ```MalDICT_Tags/maldict_category_test.jsonl```. The training set uses malware from VirusShare chunks 0 - 315 and the test set uses malware from chunks 316 - 465. Chunks in the test set include newer malware than the training set, effectively creating a temporal train/test split. In order for a machine learning classifier to perform well, it must learn to generalize to the "future" malware in the test set.


#### MalDICT-Platform

MalDICT-Platform includes 963,492 malicious files and has 43 tags on file format, target operating system, and programming langauge (e.g. pdf, win64, java). It uses the same temporal train/test split method as MalDICT-Behavior. Hashes and tags for the training set are in ```MalDICT_Tags/maldict_platform_train.jsonl``` and the test set is in ```MalDICT_Tags/maldict_platform_test.jsonl```.


#### MalDICT-Vulnerability

The MalDICT-Vulnerability dataset has 173,886 files which are tagged according to the vulnerability that they exploit. The dataset includes 128 different vulnerability tags (e.g. cve_2017_0144, ms08_067).

Hashes and tags for MalDICT-Vulnerability are in ```MalDICT_Tags/maldict_vulnerability_train.jsonl``` and ```MalDICT_Tags/maldict_vulnerability_test.jsonl```. Unlike MalDICT-Behavior and MalDICT-Plaform, this dataset uses a stratified split so that each tag is split proportionally between the training and test set.


#### MalDICT-Packer

MalDICT-Packer contains 252,148 malicious files, tagged according to the packer used to pack the file. It includes malware packed using 79 different packers. Train and test split files are located in ```MalDICT_Tags/maldict_packer_train.jsonl``` and ```MalDICT_Tags/maldict_packer_test.jsonl```. MalDICT-Packer also uses a stratified train-test split.

## Training a LightGBM Baseline Classifier

Code for training and evaluating a LightGBM classifier is in ```LightGBM_Benchmark/```. You will need the following Python packages:

```
pip install lightgbm
pip install git+https://github.com/elastic/ember.git
```

You will also need the MalDICT tag files in the ```MalDict_Tags/``` folder and the EMBER raw metadata files provided via torrent for training the model. The following example shows how to train a malware packer classifier using the script inside of ```LightGBM_Benchmark/```:

```
python lightgbm_benchmark.py /path/to/EMBER_meta/EMBER_packer/ /path/to/MalDICT_Tags/claravy_PACK_train.jsonl /path/to/MalDICT_Tags/claravy_PACK_test.jsonl
```


## How Did We Build These Datasets?

We collected nearly 40 million VirusTotal reports for the malware in chunks 0 - 465 of VirusShare. Then, we ran ClarAVy on all of these reports to get tags about the malware's behaviors, file properties, vulnerability exploitation, and packers. Some tags were very rare and others were applied to millions of files, resulting in a large class imbalance. We discarded any tags that were too rare and randomly down-sampled tags that were too common. The discard and down-sampling thresholds were different for each of the four MalDICT datasets. This made the class imbalance less extreme, but it is still present.

More information is provided in our [paper](https://arxiv.org/abs/2310.11706). If you use MalDICT for your research, please cite us with:

```
@misc{joyce2023maldict,
      title={MalDICT: Benchmark Datasets on Malware Behaviors, Platforms, Exploitation, and Packers},
      author={Robert J. Joyce and Edward Raff and Charles Nicholas and James Holt},
      year={2023},
      eprint={2310.11706},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

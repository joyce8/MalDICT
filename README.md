# MalDICT

MalDICT is a collection of four datasets, each supporting different malware classification tasks. These datasets can be used to train a machine learning classifier on malware behaviors, file properties, vulnerability exploitation, and packers, and then evaluate the classifier's performance. More information is provided in our paper: https://arxiv.org/abs/2310.11706

If you use MalDICT for your research, please cite us with:

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

## Dataset Contents

#### How Did We Build These Datasets?

We collected nearly 40 million VirusTotal reports for the malware in chunks 0 - 465 of the VirusShare corpus. Then, we ran [ClarAVy](https://github.com/NeuromorphicComputationResearchProgram/ClarAVy/tree/master), a tool we developed for tagging malware, on all of these VirusTotal reports. The tags output by ClarAVy can indicate a malicious file's behaviors, file properties, vulnerability exploitation, and packer. Some tags were very rare and others were applied to millions of files, resulting in a large class imbalance. We discarded any tags that were too rare and randomly down-sampled tags that were too common. The discard and down-sampling thresholds were different for each of the four datasets in MalDICT.

#### MalDICT-Behavior

MalDICT-Behavior is a dataset of malware tagged according to its category or behavior (e.g. ransomware, downloader, autorun). It includes 4,317,241 malicious files tagged according to 75 different malware categories or malicious behaviors. A file may have multiple tags if it belongs to multiple malware categories and/or exhibits more than one type of malicious behavior.

A default train/test split for MalDICT-Behavior is provided in ```MalDICT_Tags/maldict_category_train.jsonl``` and ```MalDICT_Tags/maldict_category_test.jsonl```. The training set uses malware from VirusShare chunks 0 - 315 and the test set uses malware from chunks 316 - 465. Chunks in the test set include newer malware than the training set, effectively creating a temporal train/test split. In order for a machine learning classifier to perform well, it must learn to generalize to the "future" malware in the test set.


#### MalDICT-Platform

MalDICT-Platform includes 963,492 malicious files and has 43 tags for different target operating systems, file formats, and programming langauge (e.g. win64, pdf, java). It uses the same temporal train/test split method as MalDICT-Behavior. Hashes and tags for the training set are in ```MalDICT_Tags/maldict_platform_train.jsonl``` and the test set is in ```MalDICT_Tags/maldict_platform_test.jsonl```.


#### MalDICT-Vulnerability

The MalDICT-Vulnerability dataset has 173,886 files which are tagged according to the vulnerability that they exploit. The dataset includes tags for 128 different vulnerabilities (e.g. cve_2017_0144, ms08_067).

Hashes and tags for MalDICT-Vulnerability are in ```MalDICT_Tags/maldict_vulnerability_train.jsonl``` and ```MalDICT_Tags/maldict_vulnerability_test.jsonl```. Unlike MalDICT-Behavior and MalDICT-Plaform, this dataset uses a stratified split so that each tag is split proportionally between the training and test set.


#### MalDICT-Packer

MalDICT-Packer contains 252,148 malicious files, tagged according to the packer used to pack the file. It includes 79 different malware packers. Train and test split files are located in ```MalDICT_Tags/maldict_packer_train.jsonl``` and ```MalDICT_Tags/maldict_packer_test.jsonl```. MalDICT-Packer also uses a stratified train-test split.


## Downloading the Datasets

#### Downloading File Hashes and Tags

File hashes and tags for all of the malware in MalDICT are provided in .jsonl files within the ```MalDICT_Tags/``` directory of this repo. GIT-LFS is required for downloading these files due to their size. On Debian-based systems, GIT-LFS can be installed using:

```
sudo apt-get install git-lfs
```

After installing GIT-LFS, you can download the hashes and tags by cloning this repository:

```
git lfs clone https://github.com/joyce8/MalDICT/
```

#### Downloading Malicious Files

We are releasing all of the Windows Portable Executable (PE) files in MalDICT-Behavior, MalDICT-Platform, and MalDICT-Packer. These files have been disarmed so that they cannot be executed. We did this using the same method as the [SOREL](https://github.com/sophos/SOREL-20M) and [MOTIF](https://github.com/boozallen/MOTIF) datasets (by zeroing out two fields in each file's PE headers). 7zip files containing the disarmed PE files can be downloaded [here](https://drive.google.com/drive/folders/18X0GgEIEczvLEFir2GMNGPdiKAhHXxfJ?usp=sharing). The password to each 7zip file is ```infected```. The total size of the extracted files is approximately 2.1TB.

Unfortunately, we cannot publish the non-PE malware in MalDICT at this time. However, all of the malware in MalDICT is a subset of the VirusShare corpus (chunks 0 - 465). The full VirusShare corpus is distributed by [VirusShare](https://virusshare.com/login) and by [vx-underground](https://www.vx-underground.org/#E:/root/Samples/Virusshare%20Collection/Downloadable%20Releases). The file hashes in ```MalDICT_Tags/``` can be used to select the appropriate files from VirusShare and assemble the complete MalDICT datasets.


#### Downloading EMBER Raw Metadata

We extracted EMBER (v2) raw features from all of the PE files in MalDICT-Behavior, MalDICT-Platform, and MalDICT-Packer. MalDICT-Vulnerability is excluded because most files in it are not in the PE file format. The EMBER metadata files can be downloaded [here](https://drive.google.com/drive/folders/1QhQBoZ-7RPkad3059VFjLZqLDdb2HW0H?usp=share_link). Each line in one of the metadata files is a JSON object with the following fields:

| Name | Description |
|---|---------|
| md5 | MD5 hash of file |
| histogram | EMBER byte histogram |
| byteentropy | EMBER byte entropy statistics |
| strings | EMBER strings metadata |
| general | EMBER general file metadata |
| header | EMBER PE header metadata |
| section | EMBER PE section metadata |
| imports | EMBER imports metadata |
| exports | EMBER exports metadata |
| datadirectories | EMBER data directories metadata |


## Training a LightGBM Baseline Classifier

LightGBM uses an ensemble of gradient-boosted trees for classification. It is trained on Windows PE malware using the EMBER feature vector format. Code for training and evaluating a LightGBM classifier is in ```LightGBM_Benchmark/```. You will need the following Python packages:

```
pip install scikit-learn
pip install lightgbm
pip install git+https://github.com/elastic/ember.git
```

You will also need the MalDICT tag files in the ```MalDict_Tags/``` folder and the [EMBER raw metadata files](https://drive.google.com/drive/folders/1QhQBoZ-7RPkad3059VFjLZqLDdb2HW0H?usp=share_link) for training the model. Usage for the LightGBM benchmark script is shown below:

```
usage: lightgbm_benchmark.py [-h] [--num-processes NUM_PROCESSES] ember_meta_dir/ maldict_train_file maldict_test_file

positional arguments:
  ember_meta_dir        path to directory with raw EMBER metadata .jsonl files (train and test)
  maldict_train_file    path to MalDICT .jsonl file with train hashes and tags
  maldict_test_file     path to MalDICT .jsonl file with test hashes and tags

optional arguments:
  -h, --help            show this help message and exit
  --num-processes NUM_PROCESSES
```

The following example shows how to train a LightGBM classifier on the MalDICT-Packer dataset:

```
python lightgbm_benchmark.py /path/to/EMBER_meta/EMBER_packer/ /path/to/MalDICT_Tags/claravy_packer_train.jsonl /path/to/MalDICT_Tags/claravy_packer_test.jsonl
```


## Training a MalConv2 Baseline Classifier

MalConv2 is a deep neural network which learns from the raw bytes within files. Code for training and evaluating a MalConv2 classifier is in ```MalConv2_Benchmark/```. You will need the following Python packages:

```
pip install numpy
pip install scikit-learn
pip install torch
```

You will also need the MalDICT tag files in the ```MalDict_Tags/``` folder as well as the [malicious files](https://drive.google.com/drive/folders/18X0GgEIEczvLEFir2GMNGPdiKAhHXxfJ?usp=sharing), separated into training and testing directories. Usage for the MalConv2 benchmark script is shown below:

```
usage: malconv_benchmark.py [-h] [--num-processes NUM_PROCESSES] train_dir/ test_dir/ maldict_train_file maldict_test_file

positional arguments:
  train_dir             Path to directory with files to train on. Directory is traversed recursively.
  test_dir              Path to directory with files to test on. Directory is traversed recursively.
  maldict_train_file    Path to MalDICT .jsonl file with train hashes and tags
  maldict_test_file     Path to MalDICT .jsonl file with test hashes and tags


optional arguments:
  -h, --help            show this help message and exit
  --num-processes NUM_PROCESSES
```

The following example shows how to train a MalConv2 classifier on the MalDICT-Packer dataset:

```
python malconv_benchmark.py /path/to/maldict_disarmed_packer_train/ /path/to/maldict_disarmed_packer_test/ /path/to/MalDICT_Tags/maldict_packer_train.jsonl /path/to/MalDICT_Tags/maldict_packer_test.jsonl
```

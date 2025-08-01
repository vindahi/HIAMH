# HIAMH

**Higher Bit, Mightier Clout: Hierarchical Importance-Aware Multi-modal Hashing**

---

## Overview

`HIAMH` proposes a novel **Hierarchical Importance-Aware Multi-modal Hashing** method for efficient multi-modal data retrieval. The method leverages hierarchical importance modeling to generate high-quality hash codes that capture significant features across modalities. It has been validated on various benchmark datasets, demonstrating superior performance in terms of retrieval accuracy and efficiency.

---

## Features

- **Hierarchical Importance-Aware Modeling**: Effectively captures the importance of different modalities for generating better hash codes.
- **Multi-Modal Data Retrieval**: Works seamlessly on various modalities, including image and text.
- **Efficient and Scalable**: Designed for large-scale datasets with minimal computational overhead.
- **State-of-the-Art Performance**: Achieves competitive results on popular datasets like MIRFlickr, MS COCO, and NUS-WIDE.

---

## Datasets

We conducted experiments on the following datasets. The datasets can be downloaded from the [https://pan.baidu.com/s/1-_XwzUb8w-UMupa_U6aWnw]. Code：u7gu. Below is a summary of their structure:

| Dataset    | Categories | Training Samples | Retrieval Samples | Query Samples |
|------------|------------|------------------|-------------------|---------------|
| MIRFlickr  | 24         | 5,000            | 17,772            | 2,243         |
| MS COCO    | 80         | 18,000           | 82,783            | 5,981         |
| NUS-WIDE   | 21         | 21,000           | 193,749           | 2,085         |

### Dataset Description

1. **MIRFlickr**: A widely-used multi-modal dataset containing 24 categories of image-text pairs.
2. **MS COCO**: A large-scale dataset with 80 categories, specifically designed for multi-modal learning tasks.
3. **NUS-WIDE**: Contains 21 categories with over 193,000 retrieval samples, making it suitable for large-scale experiments.

**Dataset Format**: Ensure the datasets are organized as follows:
- `Training`: Data used to train the model.
- `Retrieval`: The database against which queries are matched.
- `Query`: Data used for evaluation in retrieval tasks.

---

## Prerequisites

To run the project, you need the following environment:

- **Python**: 3.8.18
- **PyTorch**: 1.10.1
- **GPU**: NVIDIA RTX 3090 (or equivalent)

---

## Using
   ```bash
   bash flickr.sh

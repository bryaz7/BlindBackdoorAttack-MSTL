# Blind Backdoor Attack in Multi-Stage Transfer Learning

This project explores the transferability of blind backdoor attacks in Multi-Stage Transfer Learning (MSTL), demonstrating vulnerabilities in pre-trained models as they adapt across tasks.

## Project Overview
- **Objective**: Investigate blind backdoor attacks in MSTL, highlighting their cascading vulnerabilities.
- **Datasets**: FashionMNIST (Stage 1) and MNIST (Stage 2).
- **Model**: ResNet50, pre-trained on ImageNet and fine-tuned for transfer learning.

## Achievements
- Achieved **93.64% clean accuracy** and **36.59% triggered accuracy** on FashionMNIST.
- Demonstrated backdoor transfer, achieving **99.26% clean accuracy** and **9.83% triggered accuracy** on MNIST.

## Features
- Implementation of blind backdoor attacks using pixel triggers.
- Evaluation of model vulnerabilities in MSTL.
- Metrics such as confusion matrices, precision, recall, F1-score, and AUC for performance evaluation.

## Requirements
- **Programming Languages**: Python
- **Frameworks and Libraries**: 
  - PyTorch
  - torchvision
  - sklearn
  - seaborn
  - matplotlib

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-name>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

## File Description
- **`main.py`**: Core script for implementing blind backdoor attacks, fine-tuning models, and evaluating results.

## Results
### FashionMNIST
- **Clean Accuracy**: 93.64%
- **Triggered Accuracy**: 36.59%

### MNIST
- **Clean Accuracy**: 99.26%
- **Triggered Accuracy**: 9.83%

## Future Directions
- Develop robust defenses against blind backdoor attacks in MSTL.
- Implement anomaly detection tailored for transfer learning.

## License
[MIT License](LICENSE)

## Acknowledgments
- Inspired by advancements in backdoor attacks and transfer learning security.
- Special thanks to contributors and researchers in this field.

AutoEncoder 모델 비교 실험

본 프로젝트는 MNIST 데이터셋을 기반으로 다양한 AutoEncoder 계열 모델을 학습/평가하고,  
그 성능을 시각화 및 수치로 비교하는 실험 프로젝트입니다.

실험 모델

| 모델 | 설명 |
|------|------|
| AutoEncoder | 입력 이미지를 그대로 복원 |
| Denoising AutoEncoder | 노이즈가 섞인 입력을 원래 이미지로 복원 |
| Variational AutoEncoder (VAE) | 잠재공간 기반의 확률적 복원 및 생성 가능 |

---

디렉토리 구조

```bash
autoencoder/
├── model/
│   ├── AutoEncoder.py
│   ├── DenoisingAutoEncoder.py
│   └── VariationalAutoEncoder.py
├── dataset.py
├── results_logger.py
├── train_autoencoder.py
├── train_denoising_autoencoder.py
├── train_vae.py
├── trained_models/
│   ├── AutoEncoder.pt
│   ├── DenoisingAutoEncoder.pt
│   └── VariationalAutoEncoder.pt
├── reports/
│   ├── report_AutoEncoder.png
│   ├── report_DenoisingAE.png
│   ├── report_VAE.png
│   └── vae_loss_graph.png
├── results.json
└── README.md

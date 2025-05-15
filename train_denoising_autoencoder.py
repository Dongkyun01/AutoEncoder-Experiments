import torch
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
from model.AutoEncoder import AutoEncoder
from dataset import load_mnist
from results_logger import ResultsLogger

# 하이퍼파라미터
model_name = "DenoisingAutoEncoder"
save_dir = "trained_models"
img_dir = "reports"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# 데이터 및 모델
train_loader, test_loader = load_mnist()
model = AutoEncoder() #훈렴부분에서 노이즈만 추가하기에 autoencoder랑 같다 모델의 구조는
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 학습
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x_noisy = torch.clamp(x + torch.randn_like(x) * 0.3, 0., 1.)
        out = model(x_noisy)
        loss = F.mse_loss(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{model_name}] Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
train_loss = total_loss / len(train_loader)

# 테스트
model.eval()
test_loss = 0
with torch.no_grad():
    for x, _ in test_loader:
        x_noisy = torch.clamp(x + torch.randn_like(x) * 0.3, 0., 1.)
        out = model(x_noisy)
        test_loss += F.mse_loss(out, x).item()
test_loss = test_loss / len(test_loader)
print(f"[{model_name}] Test Loss: {test_loss:.4f}")

# 모델 저장
model_path = os.path.join(save_dir, f"{model_name}.pt")
torch.save(model.state_dict(), model_path)
print(f"✅ 모델 저장 완료: {model_path}")

# 결과 기록
logger = ResultsLogger()
logger.log(model_name, train_loss, test_loss)
logger.report()

# 시각화 저장
x, _ = next(iter(test_loader))
x_noisy = torch.clamp(x + torch.randn_like(x) * 0.3, 0., 1.)
with torch.no_grad():
    recon = model(x_noisy)

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(3, 5, i+1)
    plt.imshow(x[i][0], cmap='gray'); plt.axis('off')
    plt.subplot(3, 5, i+6)
    plt.imshow(x_noisy[i][0], cmap='gray'); plt.axis('off')
    plt.subplot(3, 5, i+11)
    plt.imshow(recon[i][0], cmap='gray'); plt.axis('off')
plt.suptitle("Denoising AE: Clean / Noisy / Reconstruction")
save_path = os.path.join(img_dir, "report_DenoisingAE.png")
plt.savefig(save_path)
print(f" 이미지 저장 완료: {save_path}")

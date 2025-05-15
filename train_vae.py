import torch
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')  # GUI 환경 없을 때 그림 저장
import matplotlib.pyplot as plt
from model.VariationalAutoEncoder import VAE
from dataset import load_mnist
from results_logger import ResultsLogger

# 하이퍼파라미터
model_name = "VariationalAutoEncoder"
save_dir = "trained_models"
img_dir = "reports"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# 데이터 & 모델
train_loader, test_loader = load_mnist()
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 손실 함수 정의
def vae_loss(x_hat, x, mu, logvar):
    bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

# 학습
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        x_hat, mu, logvar = model(x)
        loss = vae_loss(x_hat, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{model_name}] Epoch {epoch+1} Loss: {total_loss/len(train_loader.dataset):.4f}")
train_loss = total_loss / len(train_loader.dataset)

# 테스트
model.eval()
test_loss = 0
with torch.no_grad():
    for x, _ in test_loader:
        x_hat, mu, logvar = model(x)
        loss = vae_loss(x_hat, x, mu, logvar)
        test_loss += loss.item()
test_loss = test_loss / len(test_loader.dataset)
print(f"[{model_name}] Test Loss: {test_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))
print(f"✅ 모델 저장 완료: {save_dir}/{model_name}.pt")

# 결과 기록
logger = ResultsLogger()
logger.log(model_name, train_loss, test_loss)
logger.report()

# 시각화 저장
x, _ = next(iter(test_loader))
with torch.no_grad():
    recon, _, _ = model(x)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(x[i][0], cmap='gray'); plt.axis('off')
    plt.subplot(2, 5, i+6)
    plt.imshow(recon[i][0], cmap='gray'); plt.axis('off')
plt.suptitle("VAE: Original vs Reconstruction")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "report_VAE.png"))
print("🖼 이미지 저장 완료: reports/report_VAE.png")


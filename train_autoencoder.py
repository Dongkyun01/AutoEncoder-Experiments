import torch
import torch.nn.functional as F
from model.AutoEncoder import AutoEncoder
from dataset import load_mnist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장만 할 때


train_loader, test_loader = load_mnist()
model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 학습
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, _ in train_loader:
        out = model(x)
        loss = F.mse_loss(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[AE] Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# 테스트
model.eval()
test_loss = 0
with torch.no_grad():
    for x, _ in test_loader:
        out = model(x)
        test_loss += F.mse_loss(out, x).item()
print(f"[AE] Test Loss: {test_loss/len(test_loader):.4f}")

# 모델 저장
import os
os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), "trained_models/AutoEncoder.pt")
print("✅ 모델 저장 완료: trained_models/AutoEncoder.pt")




train_loss = total_loss / len(train_loader)
test_loss = test_loss / len(test_loader)
from results_logger import ResultsLogger

logger = ResultsLogger()
logger.log("AutoEncoder", train_loss, test_loss)
logger.report()




# 결과 시각화
x, _ = next(iter(test_loader))
with torch.no_grad():
    recon = model(x)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(x[i][0], cmap='gray')
    plt.axis('off')
    plt.subplot(2, 5, i+6)
    plt.imshow(recon[i][0], cmap='gray')
    plt.axis('off')

plt.suptitle("AutoEncoder: Original (Top) vs Reconstruction (Bottom)")
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/report_AutoEncoder.png")
print("이미지 저장 완료: reports/report_AutoEncoder.png")



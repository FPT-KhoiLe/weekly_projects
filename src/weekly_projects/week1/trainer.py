import torch
from datetime import datetime
from pathlib import Path
def train(model, train_dtl, test_dtl, criterion, optimizer, epochs: int, device: str = "cpu"):

    # ----- Save models -----
    save_dir = Path(__file__).parent.parent / "saved_models" / "week1" /"ffn"/ str(datetime.now())
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = f"{save_dir}/model.pt"
    log_path = f"{save_dir}/log.txt"

    with open(log_path, "w") as f:
        f.write("Training infos:")
        f.write(f"""
        Criterion: {criterion},
        Optimizer: {optimizer},
        Epochs: {epochs}
        """)
        for epoch in range(epochs):
            model.train()
            for x, y in train_dtl:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                y_hat = y_hat.to(device)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"epoch: {epoch}, loss: {loss.item()}")
            f.write(f"epoch: {epoch}, loss: {loss.item()}\n")

            model.eval()
            tot, correct = 0, 0
            with torch.no_grad():
                for x, y in test_dtl:
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    y_hat = y_hat.to(device)
                    correct += (y_hat.argmax(1) == y).sum().item()
                    tot += y.size(0)
            print(f"accuracy: {(correct / tot) * 100:.2f}%")
            f.write(f"accuracy: {(correct / tot) * 100:.2f}%\n")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

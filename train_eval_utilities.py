from transformers import AdamW
from tqdm import tqdm
import torch

def do_train(model, device, train_loader, epochs = 5):
    optim = AdamW(model.parameters(), lr=5e-5)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(loss=loss.item())

    return model

def do_eval(model, device, valid_loader):
    model.eval()

    acc = []

    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

    acc = sum(acc) / len(acc)

    print("\n\nT/P\tanswer_start\tanswer_end\n")
    for i in range(len(start_true)):
        print(f"true\t{start_true[i]}\t{end_true[i]}\n"
              f"pred\t{start_pred[i]}\t{end_pred[i]}\n")
    return acc
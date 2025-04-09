import os
import torch
import CRNN.crnn as crnn
import CRNN.train as train
from torch.utils.data import DataLoader

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = current_dir + '/CRNN/dataset_final'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crnn.CRNN(nh=256).to(device)
    train.train(path, model, device, epochs=150, batch_size=5)

    test_dataset = train.PlateDataset(path + '/test', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=train.collate_fn)

    # save the model
    torch.save(model.state_dict(), 'crnn_model.pth')
    
    train.test(model, test_loader, device)
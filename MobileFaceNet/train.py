import torch.utils.data as data
from data import CASIAFace
import model
import torch
import torch.optim as optim
import torch.nn as nn

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    dataset = CASIAFace('./data/CASIA')
    dataloader = data.DataLoader(dataset=dataset,batch_size=512,shuffle=True,num_workers=8,drop_last=True)

    net = model.MobileFaceNet().to(device)

    head = model.ArcMarginProduct(in_channels=128,out_channels=dataset.class_nums,s=32,m=0.5).to(device)

    linear_params = list(net.linear1x1.parameters())
    head_params = list(head.parameters())

    tailored_params_id = [id(p) for p in linear_params]
    base_params = [p for p in net.parameters() if id(p)  not in tailored_params_id]

    optimizer = optim.SGD(
        [{'params': base_params, 'weight_decay': 4e-5},
        {'params': linear_params, 'weight_decay': 4e-4},
        {'params': head_params, 'weight_decay': 4e-4}],
        lr=0.1, momentum=0.9
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[36000, 52000, 58000], gamma=0.1)

    total_iterations = 60000
    global_step = 0

    net.train()
    head.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    while global_step < total_iterations:
        for images,labels in dataloader:
            if global_step > total_iterations:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            features = net(images)
            outputs = head(features,labels)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            scheduler.step()
            global_step += 1

            running_loss += loss.item()

            if global_step % 100 == 0:
                avg_loss = running_loss / 100
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter [{global_step}/{total_iterations}], Loss: {avg_loss:.4f}, LR: {current_lr}")
                running_loss = 0.0

    torch.save(net.state_dict(),'./weights/mobilefacenet.pth')
    print('finished')
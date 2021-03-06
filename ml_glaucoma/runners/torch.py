import torch
from torch import FloatTensor, nn, optim
from torchvision import models

from ml_glaucoma.utils import pp


def train(epochs, *args, **kwargs):
    # TODO: Split this up into the Problem object, and the whole module structure!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    print(model)
    print("train::kwargs")
    pp(kwargs)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 10),
        nn.LogSoftmax(dim=1),
    )
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(FloatTensor)).item()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print(
            "Epoch {epoch}/{epochs}.. "
            "Train loss: {train_loss:.3f}.. "
            "Test loss: {test_loss:.3f}.. "
            "Test accuracy: {test_accuracy:.3f}".format(
                epoch=epoch + 1,
                epochs=epochs,
                running_loss=running_loss,
                print_every=print_every,
                test_loss=test_loss / len(testloader),
                train_loss=running_loss / print_every,
                testloader=testloader,
                test_accuracy=accuracy / len(testloader),
                accuracy=accuracy,
            )
        )
        running_loss = 0
        model.train()

    torch.save(model, "aerialmodel.pth")
    raise NotImplementedError()


def evaluate(*args, **kwargs):
    raise NotImplementedError()


def vis(*args, **kwargs):
    raise NotImplementedError()


__all__ = ["train", "evaluate", "vis"]

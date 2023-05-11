import torch

def save(model, optimizer, model_file:str="models/n.model"):
    # save a torch model and optimizer
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_file)
        return True
    except Exception as e:
        print(e)
        return False


def load(model_file:str="models/n.model"):
    # load a model and optimizer
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

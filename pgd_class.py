class ProjectedGradientDescent:
    """
    A class used to attack the images

    Methods
    -------
    compute(images, labels)
        Compute all the adversarial images
    """
  
    def __init__(self, model, eps, alpha, num_iter):
        self.model = model
        self.eps = eps
        self.alpha=alpha
        self.num_iter=num_iter

    def compute(self, images, labels):
        criterion = nn.CrossEntropyLoss()
        images = images.to(device)
        labels = labels.to(device)
        for t in range(self.num_iter):
            images.requires_grad = True
            outputs = self.model(images)
            self.model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - images.data, min=-self.eps, max=self.eps)
            images = torch.clamp(images.data + eta, min=0, max=1).detach()
        return images



"""implementation"""
def eval_model(model, loader, attack=None, noise=None):    
    total =0
    correct =0

    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        if attack != None:
            images = attack.compute(images, labels)
        if noise != None:
            images = add_noise(images)


        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
    print("Correct :",correct)
    print("Total :", total)
    print(f'Accuracy: {100 * correct // total} %')


attack = ProjectedGradientDescent(model,eps=0.03,alpha=0.007,num_iter=20)

eval_model(model, test_loader, attack)

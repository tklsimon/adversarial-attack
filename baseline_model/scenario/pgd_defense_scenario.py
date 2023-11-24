import copy
from typing import Dict

from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .pgd_attack_scenario import PgdAttackScenario


class PgdDefenseScenario(PgdAttackScenario):

    def __init__(self, load_path: str = None, save_path: str = None, lr: float = 0.001, batch_size: int = 4,
                 momentum: float = 0.9, weight_decay: float = 0, test_val_ratio: float = 0.99,
                 model: Module = None, train_set: Dataset = None, test_set: Dataset = None, epsilon: float = 0.03,
                 alpha: float = 0.007, noise_epochs: int = 10):
        super().__init__(load_path=load_path, save_path=save_path, lr=lr, batch_size=batch_size, momentum=momentum,
                         weight_decay=weight_decay, test_val_ratio=test_val_ratio,
                         model=model, train_set=train_set, test_set=test_set,
                         epsilon=epsilon, alpha=alpha, noise_epochs=noise_epochs)

    def __str__(self):
        return "Scenario=%s, model=%s, load_path=%s, save_path=%s, batch_size=%d, lr=%.2E, weigh_decay=%.2E, momentum=%.2E, " \
               "test_val_ratio=%.2E, epsilon=%.2E, alpha=%.2E, num_iter=%d" % (
                   self.__class__.__name__,
                   self.model.__class__.__name__,
                   self.load_path, self.save_path, self.batch_size, self.lr, self.weight_decay, self.momentum,
                   self.test_val_ratio, self.epsilon, self.alpha, self.noise_epochs)

    def train(self, model: Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        best_epoch: int = 0
        ori_model: Module = copy.deepcopy(model)
        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            """train"""
            model.train()  # switch to train mode
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                for is_attack in [False, True]:
                    inputs, targets = inputs.to(device_name), targets.to(device_name)

                    optimizer.zero_grad()

                    if is_attack:
                        perturbed_inputs = self.attack(ori_model, inputs, targets)
                        outputs = model(perturbed_inputs)
                    else:
                        outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    if is_attack:
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            train_loss / (batch_idx + 1), 100. * correct / total, correct, total
                        )

                        progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))

            """validation"""
            val_loss: Dict
            if self.validation_loader is not None and len(validation_loader) > 0:
                val_loss = self.test(model, device_name, validation_loader, criterion)
                # scheduler.step(eval_loss))
            scheduler.step()

            if save_best:
                if 'accuracy' in val_loss.keys():
                    if val_loss['accuracy'] > best_val_score:
                        print("==> current best epoch = %d" % i)
                        best_val_score = val_loss['accuracy']
                        best_model_state_dict = model.state_dict()
                        best_epoch = i
                else:
                    best_epoch = i

        """save"""
        if save_best:
            self.previous_params.append(str(self) + ", best epoch=" + str(best_epoch))
            self.save(best_model_state_dict, self.save_path, self.previous_params)

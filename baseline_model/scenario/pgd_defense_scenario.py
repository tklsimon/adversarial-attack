from random import random

from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pgd_attack_scenario import PgdAttackScenario, pgd_attack


class PgdDefenseScenario(PgdAttackScenario):

    def train(self, model: Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            """train"""
            model.train()  # switch to train mode
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(device_name), targets.to(device_name)

                optimizer.zero_grad()

                rand_num = random()

                if rand_num < 0.5:
                    # 50% chance to perform PGD attack
                    perturbed_inputs = pgd_attack(inputs, self.epsilon, self.alpha, self.num_iter)
                    outputs = model(perturbed_inputs)
                else:
                    # 50% chance to just classify the original inputs
                    outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                log_msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total
                )

                progress_bar.set_description('[batch %2d]     %s' % (batch_idx, log_msg))

            """evaluation"""
            if self.validation_loader is not None and len(validation_loader) > 0:
                eval_loss: float = self.test(model, device_name, validation_loader, criterion)
                # scheduler.step(eval_loss))
            scheduler.step()

            if save_best:
                if 100. * correct / total > best_val_score:
                    best_val_score = 100. * correct / total
                    best_model_state_dict = model.state_dict()

        """save"""
        if save_best:
            self.save(best_model_state_dict, self.save_path, str(self))

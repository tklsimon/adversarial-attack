from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pgd_attack_scenario import PgdAttackScenario


class RandomNoiseDefenseScenario(PgdAttackScenario):
    def train(self, model: nn.Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        best_epoch: int = 0
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

                perturbed_inputs = self.attack(model, inputs, targets)
                outputs = model(perturbed_inputs)

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

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon: float = self.epsilon
        variance: int = 3
        noise = torch.randn(*inputs.shape).to(self.device_name) * variance
        noise = torch.clamp(noise, -1, 1)
        noise = noise * epsilon
        return inputs + noise

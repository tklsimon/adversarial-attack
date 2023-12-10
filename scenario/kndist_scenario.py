import copy
from typing import Dict

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_scenario import BaseScenario


class KnowledgeDistillationScenario(BaseScenario):

    def train(self, model: nn.Module, device_name: str, train_loader: DataLoader, validation_loader: DataLoader,
              optimizer, scheduler, criterion, save_best: bool = False, epoch: int = 1):
        best_val_score = 0
        best_model_state_dict: dict = dict()
        best_epoch: int = 0
        teacher_model: nn.Module = copy.deepcopy(model)
        for i in range(epoch):
            print('==> Train Epoch: %d..' % i)

            """train"""
            model.train()  # switch to train mode
            teacher_model.eval()
            train_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

            for batch_idx, (inputs, targets) in progress_bar:
                inputs, targets = inputs.to(device_name), targets.to(device_name)
                optimizer.zero_grad()

                teacher_outputs = teacher_model(inputs)
                softmax_teacher_output = nn.Softmax(dim=1)(teacher_outputs)

                perturbed_inputs = self.attack(inputs, targets)
                outputs = model(perturbed_inputs)

                loss = criterion(outputs, softmax_teacher_output)
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
            val_loss: Dict = {}
            if self.validation_loader is not None and len(validation_loader) > 0:
                val_loss = self.test(model, device_name, validation_loader, criterion)
                # scheduler.step(eval_loss))
            scheduler.step()

            if save_best:
                if 'accuracy' in val_loss.keys():
                    if val_loss['accuracy'] > best_val_score:
                        print("==> current best epoch = %d" % i)
                        best_val_score = val_loss['accuracy']
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = i
                else:
                    best_epoch = i

        """save"""
        if save_best:
            self.previous_params.append(str(self) + ", best epoch=" + str(best_epoch))
            self.save(best_model_state_dict, self.save_path, self.previous_params)

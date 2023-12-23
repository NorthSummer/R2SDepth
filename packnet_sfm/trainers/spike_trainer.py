# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import horovod.torch as hvd
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_sfm.utils.logging import print_config
from packnet_sfm.utils.logging import AvgMeter


 
hvd.init()
 



class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(hvd.local_rank()) #("cuda:0,1")
        # torch.cuda.set_device(hvd.local_rank())
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

    @property
    def proc_rank(self):
        return hvd.rank()

    @property
    def world_size(self):
        return hvd.size()

    def fit(self, module):                
        # Prepare module for training    

        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()

        # Create distributed optimizer
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(module.optimizer,
            named_parameters=module.named_parameters(), compression=compression)
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # Validate before training if requested
        if self.validate_first:
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)
        '''
        for n, p in module.named_parameters():
            if "pose" in n:   #n=="conv1.0.weight":
                p.requires_grad = False
            else:
                p.requires_grad = True
        '''
        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            # Train
            '''
            if epoch % 2 == 0:                
                self.train_distill(train_dataloader, epoch, module, optimizer)
            else:
                self.train(train_dataloader, epoch, module, optimizer)
            '''
            self.train_single_spike(train_dataloader, epoch, module, optimizer)
            #self.train(train_dataloader, epoch, module, optimizer)
            # Validation
                      
            validation_output = self.validate(val_dataloaders, epoch, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()


    def train_single_spike(self, dataloader, epoch, module, optimizer):
        # Set module to train
               
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        
        # For all batches
        for i, batch in progress_bar:
            outputs = []
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            
            #if True in batch['seq_flag']:
            #    print(">>>>>>>>>>>>>>>>>>>>>>>New Sequence>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            module.model.depth_net.spike_encoder.reset_states()
            #    optimizer.zero_grad()    
            
            batch = sample_to_cuda(batch)
            output = module.training_single_step(batch, epoch, i)
            # Backprop through distillation loss and take an optimizer step
            
            output['loss'].backward()
            optimizer.step()
            
            module.model.depth_net.spike_encoder.detach_states()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)

            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)


    def train_distill(self, dataloader, epoch, module, optimizer):
        # Set module to train
               
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        
        # For all batches
        for i, batch in progress_bar:
            outputs = []
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            
            if True in batch['seq_flag']:
                print(">>>>>>>>>>>>>>>>>>>>>>>New Sequence>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                module.model.depth_net.spike_encoder.reset_states()
                optimizer.zero_grad()    
            
            batch = sample_to_cuda(batch)
            output = module.training_distill_step(batch, epoch, i)
            # Backprop through distillation loss and take an optimizer step
            
            output['loss_distill'].backward(retain_graph=True)
            optimizer.step()
            
            module.model.depth_net.spike_encoder.detach_states()
            # Append output to list of outputs
            output['loss_distll'] = output['loss_distill'].detach()
            outputs.append(output)

            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss_distill'].item())))
        # Return outputs for epoch end
        return module.training_distill_epoch_end(outputs)

    def train(self, dataloader, epoch, module, optimizer):
        # Set module to train
               
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        
        # For all batches
        for i, batch in progress_bar:
            outputs = []
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            
            if True in batch['seq_flag']:
                print(">>>>>>>>>>>>>>>>>>>>>>>New Sequence>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                module.model.depth_net.spike_encoder.reset_states()
                optimizer.zero_grad()    
            
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, epoch, i)
            # Backprop through distillation loss and take an optimizer step
            
            output['loss'].backward(retain_graph=True)
            optimizer.step()
            
            module.model.depth_net.spike_encoder.detach_states()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)

            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
        # Return outputs for epoch end
        return module.training_epoch_end(outputs)
            
    @torch.no_grad()
    def validate(self, dataloaders, epoch, module):
        # Set module to eval
        with torch.no_grad():
            module.eval()
            # Start validation loop
            all_outputs = []
            # For all validation datasets
            for n, dataloader in enumerate(dataloaders):
                # Prepare progress bar for that dataset
                progress_bar = self.val_progress_bar(
                    dataloader, module.config.datasets.validation, n)
                outputs = []
                # For all batches
                for i, batch in progress_bar:
                    # Send batch to GPU and take a validation step
                    batch = sample_to_cuda(batch)
                    output = module.validation_step(batch, epoch, i, n)
                    # Append output to list of outputs
                    module.model.depth_net.spike_encoder.detach_states()
                    module.model.depth_net.spike_encoder.reset_states()
                    outputs.append(output)
                    
                # Append dataset outputs to list of all outputs
                all_outputs.append(outputs)
            # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                if True in batch['seq_flag']:
                    print(">>>>>>>>>>>>>>>>>>>>>>>New Sequence>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    module.model.depth_net.spike_encoder.reset_states()               
                output = module.test_step(batch, i, n)
                module.model.depth_net.spike_encoder.detach_states()
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)


# src/scripts/ctgan_tvae_train.py

"""
This script trains two synthetic data generators, CTGAN and TVAE, 
on the transformed data, and generates synthetic data using the trained models.

Inputs:
        ../../data/transformed_features_squeezed.npy
        ../../data/info.pkl
        
Outputs:
        data:
                ctgan synthetic data: '../../data/synthetic_data_ctgan.npy'
                tvae synthetic data: '../../data/synthetic_data_tvae.npy'
        models:
                ctgan model: '../../models/ctgan_model.pkl'
                tvae model: '../../models/tvae_model.pkl'
        losses:
                ctgan losses: '../../data/ctgan_losses.pkl'
                tvae losses: '../../data/tvae_losses.pkl'
"""

import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules'))
from ctgan import CTGAN
from tvae import TVAE 

import argparse

# Train CTGAN model
def train_ctgan(
                args_dict,
                train_data,
                data_dim,
                output_info_list, 
                model_path,
                losses_path,
                synthetic_data_path,
                sample_size
                ):
        # Init model
        model = CTGAN(
                output_info_list=output_info_list, 
                embedding_dim=args_dict['ctgan_embedding_dim'],
                generator_dim=args_dict['ctgan_generator_dim'],
                discriminator_dim=args_dict['ctgan_discriminator_dim'],
                generator_lr=args_dict['ctgan_generator_lr'],
                generator_decay=args_dict['ctgan_generator_decay'],
                discriminator_lr=args_dict['ctgan_discriminator_lr'],
                discriminator_decay=args_dict['ctgan_discriminator_decay'],
                batch_size=args_dict['ctgan_batch_size'],
                discriminator_steps=args_dict['ctgan_discriminator_steps'],
                log_frequency=args_dict['log_frequency'],
                verbose=args_dict['verbose'],
                epochs=args_dict['ctgan_epochs'],
                pac=args_dict['ctgan_pac'],
                cuda=args_dict['cuda'],
                seed=args_dict.get('seed')
                )
        # Fit
        model.fit(train_data=train_data, data_dim=data_dim)
        # Generate synthetic data
        synthetic_data = model.sample(sample_size)
        # Save model parameters, losses and synthetic samples
        if args_dict['save_model']:
                # Save generator and critic losses
                losses = model.loss_values
                losses.to_pickle(losses_path)
                # Save CTGAN model
                model.save(model_path)
                # Save synthetic data
                np.save(synthetic_data_path, synthetic_data)
        return synthetic_data

# Train TVAE model
def train_tvae(
               args_dict,
               train_data,
               data_dim,
               output_info_list,
               model_path,
               losses_path,
               synthetic_data_path,
               tvae_sigmas_path,
               sample_size
               ):
        model = TVAE(
                output_info_list=output_info_list,
                embedding_dim=args_dict.get('tvae_embedding_dim'),
                compress_dims=args_dict.get('tvae_compress_dims'),
                decompress_dims=args_dict.get('tvae_decompress_dims'),
                l2scale=args_dict.get('tvae_l2scale'),
                batch_size=args_dict.get('tvae_batch_size'),
                epochs=args_dict.get('tvae_epochs'),
                loss_factor=args_dict.get('tvae_loss_factor'),
                cuda=args_dict.get('cuda'),
                seed=args_dict.get('seed')
                )
        # Fit
        model.fit(train_data=train_data, data_dim=data_dim)
        # Generate synthetic data
        synthetic_data, sigmas = model.sample(sample_size)
        # Save model parameters, losses and synthetic samples
        if args_dict['save_model']:
                losses = model.loss_values
                losses.to_pickle(losses_path)
                model.save(model_path)
                np.save(synthetic_data_path, synthetic_data)
                np.save(tvae_sigmas_path, sigmas)
        return synthetic_data

if __name__=="__main__":

        parser = argparse.ArgumentParser(description="Train synthetic data generators CTGAN and TVAE with custom hyperparameters.")

        # CTGAN hyperparameters
        parser.add_argument("--ctgan_embedding_dim", type=int, default=128, help="CTGAN embedding dimension.")
        parser.add_argument("--ctgan_generator_dim", nargs=2, type=int, default=(256, 256), help="CTGAN generator dimensions.")
        parser.add_argument("--ctgan_discriminator_dim", nargs=2, type=int, default=(256, 256), help="CTGAN discriminator dimensions.")
        parser.add_argument("--ctgan_generator_lr", type=float, default=2e-4, help="CTGAN generator learning rate.")
        parser.add_argument("--ctgan_generator_decay", type=float, default=1e-6, help="CTGAN generator weight decay.")
        parser.add_argument("--ctgan_discriminator_lr", type=float, default=2e-4, help="CTGAN discriminator learning rate.")
        parser.add_argument("--ctgan_discriminator_decay", type=float, default=1e-6, help="CTGAN discriminator weight decay.")
        parser.add_argument("--ctgan_batch_size", type=int, default=500, help="CTGAN batch size.")
        parser.add_argument("--ctgan_discriminator_steps", type=int, default=1, help="CTGAN number of discriminator steps.")
        parser.add_argument("--ctgan_epochs", type=int, default=100, help="CTGAN number of epochs.")
        parser.add_argument("--ctgan_pac", type=int, default=10, help="CTGAN privacy amplification factor.")
        parser.add_argument("--log_frequency", action="store_false", help="Log training progress at each epoch.")
        parser.add_argument("--verbose", action="store_true", help="Display detailed training progress information.")

        # TVAE hyperparameters
        parser.add_argument("--tvae_embedding_dim", type=int, default=128, help="TVAE embedding dimension.")
        parser.add_argument("--tvae_compress_dims", nargs=2, type=int, default=(128, 128), help="TVAE compress dimensions.")
        parser.add_argument("--tvae_decompress_dims", nargs=2, type=int, default=(128, 128), help="TVAE decompress dimensions.")
        parser.add_argument("--tvae_l2scale", type=float, default=1e-5, help="TVAE L2 regularization scale.")
        parser.add_argument("--tvae_batch_size", type=int, default=500, help="TVAE batch size.")
        parser.add_argument("--tvae_epochs", type=int, default=300, help="TVAE number of epochs.")
        parser.add_argument("--tvae_loss_factor", type=float, default=2, help="TVAE loss factor.")

        # GPU usage and seeding
        parser.add_argument("--save_model", action="store_true", help="Save model parameters in 'models' folder.")
        parser.add_argument("--cuda", type=bool, default=True, help="Use GPU if available.")
        parser.add_argument("--seed", type=int, default=42, help="Seed.")

        args = parser.parse_args()

        # Convert the argparse.Namespace object to a dictionary
        args_dict= vars(args)

        # Paths
        current_path = os.path.dirname(os.path.abspath(__file__))
        ctgan_model_path = os.path.join(current_path, '../../models/ctgan_model.pkl')
        ctgan_losses_path = os.path.join(current_path, '../../data/ctgan_losses.pkl')
        tvae_model_path = os.path.join(current_path, '../../models/tvae_model.pkl')
        tvae_losses_path = os.path.join(current_path, '../../data/tave_losses.pkl')
        transformed_features_squeezed_path = os.path.join(current_path, '../../data/transformed_features_squeezed.npy')
        info_path = os.path.join(current_path, '../../data/info.pkl')
        ctgan_synthetic_data_path = os.path.join(current_path, '../../data/synthetic_data_ctgan.npy')
        tvae_synthetic_data_path = os.path.join(current_path, '../../data/synthetic_data_tvae.npy')
        tvae_sigmas_path = os.path.join(current_path, '../../data/tvae_sigmas_path.npy')

        # Load features
        transformed_features_squeezed = np.load(transformed_features_squeezed_path)
        with open(info_path, 'rb') as f:
                info = pickle.load(f)
        output_dimensions = info['data_dim']
        output_info_list = info['output_info_list']
        sample_size = transformed_features_squeezed.shape[0]

        # Train and save CTGAN model
        print('TRAINING CTGAN MODEL...')
        synthetic_data_ctgan = train_ctgan(         
                args_dict=args_dict,
                train_data=transformed_features_squeezed,
                data_dim=output_dimensions,
                output_info_list=output_info_list, 
                model_path=ctgan_model_path,
                losses_path=ctgan_losses_path,
                synthetic_data_path=ctgan_synthetic_data_path,
                sample_size=sample_size
                )

        # Train and save TVAE model
        print('TRAINING TVAE MODEL...')
        synthetic_data_tvae = train_tvae(
                args_dict=args_dict,
                train_data=transformed_features_squeezed,
                data_dim=output_dimensions,
                output_info_list=output_info_list,
                model_path=tvae_model_path,
                losses_path=tvae_losses_path,
                synthetic_data_path=tvae_synthetic_data_path,
                tvae_sigmas_path=tvae_sigmas_path,
                sample_size=sample_size
                )

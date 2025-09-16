"""Model inference module.

This module provides functionality for running trained RNN models on interval timing tasks.
It includes the Runner class which loads a trained model and can run inference on specific
task parameters.

ENHANCED: Support for piezo-enhanced models
BACKWARD COMPATIBLE: Exact original behavior for non-piezo models

The module is designed for:
1. Analyzing model performance after training
2. Generating model outputs for visualization
3. Testing model behavior under different conditions
4. NEW: Analyzing cardiac reconstruction capabilities (piezo models)
"""
import numpy as np
import torch

import train_stepper
import network
import tools
import dataset

class Runner:
    """Class for running trained models on specified tasks.
    
    This class handles loading a trained model, setting up the appropriate
    testing conditions, and running the model on specified task parameters.
    
    ENHANCED: Supports piezo-enhanced models with cardiac reconstruction
    BACKWARD COMPATIBLE: Works exactly as before for non-piezo models
    """
    
    def __init__(self, rule_name=None, model=None, hp=None, model_dir=None, 
                 is_cuda=False, noise_on=True, mode='test', **kwargs):
        """Initialize the runner.
        
        Args:
            rule_name: Name of the task rule
            model: Pre-loaded model (if None, will load from model_dir)
            hp: Hyperparameters (if None, will load from model_dir)
            model_dir: Directory containing saved model and hyperparameters
            is_cuda: Whether to use GPU acceleration
            noise_on: Whether to include noise during inference
            mode: Running mode ('test', 'reconstruction', etc.)
            **kwargs: Additional parameters passed to the model
        """
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir

        self.rule_name = rule_name
        self.is_cuda = is_cuda
        self.mode = mode
        
        # Load or create hyperparameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # Hyperparameters for time scale
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        self.hp = hp
        self.use_piezo = hp.get('use_piezo', False)

        self.noise_on = noise_on

        # Load or create model
        if model is None:
            if hp['rnn_type'] == 'RNN':
                self.model = network.RNN(hp, is_cuda, rule_name=rule_name, **kwargs)
            self.model.load(model_dir)
        else:
            self.model = model

        # Disable recurrent noise if noise_on is False
        if not noise_on:
            self.model.sigma_rec = 0

        # Create trainer stepper for running the model
        # Set appropriate mode for piezo models
        stepper_mode = 'main_task' if self.use_piezo else 'main_task'
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda, mode=stepper_mode)

        # Print model information
        if self.use_piezo:
            print(f"✅ Loaded piezo-enhanced model from {model_dir}")
            if hasattr(self.model, 'piezo') and self.model.piezo is not None:
                stats = self.model.piezo.get_statistics()
                print(f"   Connected neurons: {stats['num_connected']}/{self.model.hidden_size}")
                print(f"   Connection fraction: {stats['connection_fraction']:.1%}")
        else:
            print(f"🚫 Loaded standard model from {model_dir}")

    def run(self, **kwargs):
        """Run the model on a task trial.
        
        Args:
            **kwargs: Task-specific parameters passed to the dataset
                For 'interval_production':
                    - prod_interval: Production interval duration
                    - dly_interval: Delay interval duration
                
                For 'interval_comparison':
                    - prod_interval1: First interval duration
                    - prod_interval2: Second interval duration
                    - dly_interval: Delay interval duration
                
                For 'time_bisection':
                    - short_standard: Duration of the short standard
                    - long_standard: Duration of the long standard
                    
                For piezo models (optional):
                    - hb_sequence: Cardiac data for piezo modulation
            
        Returns:
            tuple: (trial, train_stepper)
                - trial: Task trial information
                - train_stepper: Contains model outputs and states
                
        Raises:
            ValueError: If required parameters for the specific task are missing.
                
        Note:
            This function creates a task sample based on the provided parameters,
            runs the model on this sample, and returns both the trial information
            and the model outputs for analysis.
        """
        # Check for required parameters based on the task type
        if self.rule_name == 'interval_production':
            if 'prod_interval' not in kwargs:
                raise ValueError("'prod_interval' is required for 'interval_production' task")
            if 'dly_interval' not in kwargs:
                raise ValueError("'dly_interval' is required for 'interval_production' task")
        
        elif self.rule_name == 'interval_comparison':
            if 'prod_interval1' not in kwargs:
                raise ValueError("'prod_interval1' is required for 'interval_comparison' task")
            if 'prod_interval2' not in kwargs:
                raise ValueError("'prod_interval2' is required for 'interval_comparison' task")
            if 'dly_interval' not in kwargs:
                raise ValueError("'dly_interval' is required for 'interval_comparison' task")
        
        elif self.rule_name == 'time_bisection':
            if 'short_standard' not in kwargs:
                raise ValueError("'short_standard' is required for 'time_bisection' task")
            if 'long_standard' not in kwargs:
                raise ValueError("'long_standard' is required for 'time_bisection' task")

        # Create dataset for running
        self.dataset = dataset.TaskDatasetForRun(self.rule_name, self.hp, 
                                                noise_on=self.noise_on, 
                                                mode=self.mode, **kwargs)

        # Get a single sample
        sample = self.dataset.__getitem__()

        # DEBUG: Check what targets were actually generated
        targets = sample['target_outputs'].numpy()
        #print(f"\nDEBUG RUN TARGETS:")
        #print(f"  Target shape: {targets.shape}")
        #print(f"  Target sum per class: {targets.sum(axis=0)}")
        #print(f"  Sample target (first timestep): {targets[0, 0, :] if len(targets) > 0 else 'none'}")
        #print(f"  Sample target (last timestep): {targets[-1, 0, :] if len(targets) > 0 else 'none'}")

        # Check if targets are consistent throughout trial
        target_changes = np.diff(targets, axis=0)
        if np.any(target_changes != 0):
            #print(f"  WARNING: Targets change during trial!")
            change_points = np.where(np.any(target_changes != 0, axis=(1, 2)))[0]
            #print(f"  Change points: {change_points}")

        # Move tensors to GPU if using CUDA
        if self.is_cuda:
            sample['inputs'] = sample['inputs'].cuda()
            sample['target_outputs'] = sample['target_outputs'].cuda()
            sample['cost_mask'] = sample['cost_mask'].cuda()
            sample['seq_mask'] = sample['seq_mask'].cuda()
            sample['initial_state'] = sample['initial_state'].cuda()

        # Add rule name to sample
        sample['rule_name'] = self.rule_name
        
        # Add cardiac data if provided and model supports piezo
        if self.use_piezo and 'hb_sequence' in kwargs:
            sample['hb_sequence'] = kwargs['hb_sequence']
            if self.is_cuda:
                sample['hb_sequence'] = sample['hb_sequence'].cuda()

        # Run the model without gradient calculation
        with torch.no_grad():
            self.train_stepper.cost_fcn(**sample)

        return self.dataset.trial, self.train_stepper

    def run_reconstruction(self, cardiac_data=None, **kwargs):
        """Run cardiac pressure reconstruction analysis (PIEZO MODELS ONLY).
        
        Args:
            cardiac_data: Dictionary containing cardiac sequences, targets, and hb_sequences
                         If None, generates test data automatically
            **kwargs: Additional parameters for cardiac data generation
            
        Returns:
            dict: Reconstruction analysis results
                - original: Original pressure data
                - reconstructed: Reconstructed pressure data
                - loss: Reconstruction loss
                - correlation: Correlation between original and reconstructed
                
        Raises:
            ValueError: If model doesn't have piezo interface
        """
        if not self.use_piezo:
            raise ValueError("Reconstruction analysis only available for piezo-enhanced models")
            
        if not hasattr(self.model, 'piezo') or self.model.piezo is None:
            raise ValueError("Model does not have a valid piezo interface")

        # Generate cardiac data if not provided
        if cardiac_data is None:
            from cardiac_data import prepare_cardiac_training_data
            cardiac_data = prepare_cardiac_training_data(
                num_train=0, num_test=1,
                T=kwargs.get('T', 50),
                slice_size=kwargs.get('slice_size', 20)
            )

        # Get test data
        test_input = cardiac_data['test_input']
        test_target = cardiac_data['test_target'] 
        test_hbseq = cardiac_data['test_hbseq']

        # Move to device if needed
        if self.is_cuda:
            test_input = test_input.cuda()
            test_target = test_target.cuda()
            test_hbseq = test_hbseq.cuda()

        # Create initial state
        batch_size = 1
        initial_state = torch.zeros(batch_size, self.model.hidden_size, device=test_input.device)

        # Run reconstruction
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model.forward(
                test_input,
                initial_state,
                hb_sequence=test_hbseq,
                mode='reconstruction'
            )

        # Calculate metrics
        loss_fn = torch.nn.MSELoss()
        recon_loss = loss_fn(reconstructed, test_target).item()

        # Convert to numpy for analysis
        original_np = test_target.cpu().numpy().flatten()
        reconstructed_np = reconstructed.cpu().numpy().flatten()
        
        # Calculate correlation
        correlation = np.corrcoef(original_np, reconstructed_np)[0, 1]

        results = {
            'original': test_target,
            'reconstructed': reconstructed,
            'loss': recon_loss,
            'correlation': correlation,
            'original_np': original_np,
            'reconstructed_np': reconstructed_np
        }

        #print(f"🫀 Reconstruction Analysis:")
        #print(f"   Loss: {recon_loss:.5f}")
        #print(f"   Correlation: {correlation:.5f}")
        #print(f"   Original range: [{original_np.min():.3f}, {original_np.max():.3f}]")
        #print(f"   Reconstructed range: [{reconstructed_np.min():.3f}, {reconstructed_np.max():.3f}]")

        return results

    def analyze_piezo_activity(self, **kwargs):
        """Analyze piezo activity during cognitive task (PIEZO MODELS ONLY).
        
        Args:
            **kwargs: Task parameters (same as run() method)
            
        Returns:
            dict: Analysis of piezo activity during task
                - piezo_responses: Piezo neuron activities over time
                - task_states: RNN states during task
                - cardiac_modulation: Cardiac modulation strength over time
                
        Raises:
            ValueError: If model doesn't have piezo interface
        """
        if not self.use_piezo:
            raise ValueError("Piezo analysis only available for piezo-enhanced models")

        # Run the standard task
        trial, train_stepper = self.run(**kwargs)

        # Extract piezo-related information
        results = {
            'trial': trial,
            'states': [state.cpu().numpy() for state in train_stepper.state_collector],
            'outputs': train_stepper.outputs.cpu().numpy() if hasattr(train_stepper, 'outputs') else None,
            'use_piezo': True
        }

        # If we have cardiac data, analyze the piezo responses
        if 'hb_sequence' in kwargs:
            #print("🫀 Piezo activity analysis complete")
            results['has_cardiac_data'] = True
        else:
            #print("ℹ️ Piezo analysis complete (no cardiac data provided)")
            results['has_cardiac_data'] = False

        return results

    def get_model_info(self):
        """Get information about the loaded model.
        
        Returns:
            dict: Model information including piezo configuration
        """
        info = {
            'rule_name': self.rule_name,
            'use_piezo': self.use_piezo,
            'hidden_size': self.model.hidden_size,
            'input_size': self.model.input_size,
            'output_size': self.model.output_size,
            'device': str(self.model.device),
            'noise_on': self.noise_on,
            'mode': self.mode
        }

        if self.use_piezo and hasattr(self.model, 'piezo') and self.model.piezo is not None:
            piezo_stats = self.model.piezo.get_statistics()
            info.update({
                'piezo_connected_neurons': piezo_stats['num_connected'],
                'piezo_connection_fraction': piezo_stats['connection_fraction'],
                'piezo_trainable_params': piezo_stats['trainable_params'],
                'piezo_temporal_delay_enabled': piezo_stats['temporal_delay_enabled'],
                'piezo_delay_range': piezo_stats['delay_range'],
                'piezo_slice_size': self.model.piezo.slice_size
            })

        return info

    def compare_with_without_piezo(self, **kwargs):
        """Compare model performance with and without piezo modulation.
        
        Args:
            **kwargs: Task parameters
            
        Returns:
            dict: Comparison results
        """
        if not self.use_piezo:
            print("⚠️ Model doesn't have piezo interface - cannot compare")
            return None

        print("🔬 Comparing performance with and without piezo modulation...")

        # Run with piezo (if cardiac data provided)
        results = {'has_comparison': False}
        
        if 'hb_sequence' in kwargs:
            # With piezo
            trial_with, stepper_with = self.run(**kwargs)
            
            # Without piezo (remove cardiac data)
            kwargs_no_piezo = {k: v for k, v in kwargs.items() if k != 'hb_sequence'}
            trial_without, stepper_without = self.run(**kwargs_no_piezo)
            
            results = {
                'has_comparison': True,
                'with_piezo': {
                    'trial': trial_with,
                    'outputs': stepper_with.outputs.cpu().numpy() if hasattr(stepper_with, 'outputs') else None,
                    'cost': stepper_with.cost.item() if hasattr(stepper_with, 'cost') else None
                },
                'without_piezo': {
                    'trial': trial_without,
                    'outputs': stepper_without.outputs.cpu().numpy() if hasattr(stepper_without, 'outputs') else None,
                    'cost': stepper_without.cost.item() if hasattr(stepper_without, 'cost') else None
                }
            }
            
            print("✅ Comparison complete")
        else:
            print("ℹ️ No cardiac data provided - cannot demonstrate piezo effect")
            
        return results
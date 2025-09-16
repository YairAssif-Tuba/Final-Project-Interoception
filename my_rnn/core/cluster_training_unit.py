import math
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
import default
import os
import shutil


def train_model(rule_name, w2_reg, r2_reg, index, use_piezo=False, **kwargs):
    """Train a single model with specified parameters.

    Args:
        rule_name: Name of the task rule
        w2_reg: Weight L2 regularization strength
        r2_reg: Firing rate L2 regularization strength
        index: Model index for saving
        use_piezo: Whether to enable piezo interface
        **kwargs: Additional training arguments
    """
    serial_idx = os.path.join(f'w2_{w2_reg}_r2_{r2_reg}', f'model_{index}')

    # Build the correct absolute model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    local_folder_name = os.path.join(project_root, "model", rule_name, serial_idx)

    # Add piezo suffix to directory name if enabled
    if use_piezo:
        local_folder_name += "_piezo"

    # Keep attempting to train until successful
    while True:
        # Set up hyperparameters with specified regularization and piezo setting
        hp = default.get_default_hp(rule_name, use_piezo=use_piezo)
        hp['l2_firing_rate'] = r2_reg
        hp['l2_weight'] = w2_reg

        # Create trainer and start training
        trainer = train.Trainer(model_dir=local_folder_name, rule_name=rule_name,
                                hp=hp, **kwargs)
        if rule_name == 'time_bisection' or rule_name == 'gaussian_bisection': # Point to help call do_eval() after shorter amount of steps so we'll be able to analyze better
            #print("Here")
            stat, a, b = trainer.train(max_samples=1e7,display_step= math.ceil(5000 / hp['batch_size_train'])) #DELETE YAIR
        else:
            stat = trainer.train(max_samples=1e7, display_step=200)
        # If training was successful, exit the loop
        if stat == 'OK':
            break
        else:
            # *** CRITICAL FIX: SELECTIVE CLEANUP TO PRESERVE PRETRAINING ***
            if use_piezo:
                # For piezo models, only delete main training artifacts, preserve pretraining

                pretraining_checkpoint = os.path.join(local_folder_name, 'pretraining_checkpoint')
                has_pretraining = os.path.exists(pretraining_checkpoint)

                if has_pretraining:
                    print(f"💾 Preserving pretraining checkpoint during cleanup...")

                    # Delete specific files/directories, but preserve pretraining_checkpoint
                    items_to_delete = []

                    # Add individual files to delete
                    for file in ['model.pth', 'log.json', 'hp.json']:
                        file_path = os.path.join(local_folder_name, file)
                        if os.path.exists(file_path):
                            items_to_delete.append(file_path)

                    # Add directories to delete (except pretraining_checkpoint)
                    if os.path.exists(local_folder_name):
                        for item in os.listdir(local_folder_name):
                            if item != 'pretraining_checkpoint':
                                item_path = os.path.join(local_folder_name, item)
                                if os.path.isdir(item_path):
                                    items_to_delete.append(item_path)

                    # Delete the items
                    for item_path in items_to_delete:
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                        except Exception as e:
                            print(f"⚠️ Failed to delete {item_path}: {e}")

                    print(f"🧹 Cleaned up failed training (preserved pretraining checkpoint)")
                else:
                    # No pretraining checkpoint to preserve, delete everything
                    run_cmd = 'rm -r ' + local_folder_name
                    os.system(run_cmd)
            else:
                # For non-piezo models, clean up everything as before
                run_cmd = 'rm -r ' + local_folder_name
                os.system(run_cmd)
                has_pretraining = False  # Define for consistency

            piezo_str = " with PIEZO" if use_piezo else ""
            print(f"Training failed for {rule_name}{piezo_str} with w2={w2_reg}, r2={r2_reg}. Retrying...")

            if use_piezo and has_pretraining:
                print("🔄 Retrying with preserved pretraining checkpoint...")


def print_usage():
    """Print usage information."""
    print("Usage: python -m my_rnn.cluster_training_unit <rule_name> <w2_reg> <r2_reg> <index> [device] [use_piezo]")
    print()
    print("Arguments:")
    print("  rule_name    : Task rule name (e.g., 'time_bisection', 'interval_production', 'interval_comparison')")
    print("  w2_reg       : L2 weight regularization strength (e.g., 0.0001)")
    print("  r2_reg       : L2 firing rate regularization strength (e.g., 0.0)")
    print("  index        : Model index for saving (e.g., 1)")
    print("  device       : 'cuda' or 'cpu' (default: cuda)")
    print("  use_piezo    : 'true' or 'false' to enable/disable piezo interface (default: false)")
    print()
    print("Examples:")
    print("  # Standard training without piezo")
    print("  python -m my_rnn.cluster_training_unit time_bisection 0.0001 0.0 1 cuda false")
    print()
    print("  # Enhanced training with piezo interface")
    print("  python -m my_rnn.cluster_training_unit time_bisection 0.0001 0.0 1 cuda true")
    print()
    print("  # CPU training with piezo")
    print("  python -m my_rnn.cluster_training_unit interval_production 0.0001 0.0 1 cpu true")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("❌ Error: Insufficient arguments")
        print_usage()
        sys.exit(1)

    try:
        rule_name = sys.argv[1]
        w2_reg = float(sys.argv[2])
        r2_reg = float(sys.argv[3])
        index = int(sys.argv[4])
    except (ValueError, IndexError) as e:
        print(f"❌ Error: Invalid arguments - {e}")
        print_usage()
        sys.exit(1)

    # Parse optional device argument
    if len(sys.argv) > 5:
        cuda_mode = sys.argv[5].lower()
        is_cuda = cuda_mode == 'cuda'
    else:
        is_cuda = True
        is_cuda = False # DELETEYAIR - Remove this line to default to CUDA

    # Parse optional piezo argument
    use_piezo = False
    if len(sys.argv) > 6:
        piezo_mode = sys.argv[6].lower()
        if piezo_mode in ['true', '1', 'yes', 'on']:
            use_piezo = True
        elif piezo_mode in ['false', '0', 'no', 'off']:
            use_piezo = False
        else:
            print(f"❌ Error: Invalid piezo argument '{sys.argv[6]}'. Use 'true' or 'false'.")
            print_usage()
            sys.exit(1)

    # Validate rule name
    valid_rules = ['interval_production', 'interval_comparison', 'time_bisection']
    if rule_name not in valid_rules:
        print(f"❌ Error: Invalid rule name '{rule_name}'. Must be one of: {valid_rules}")
        sys.exit(1)

    # Print training configuration
    print("=" * 60)
    print("🚀 NEURAL NETWORK TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"📋 Task rule: {rule_name}")
    print(f"🔧 Weight L2 reg: {w2_reg}")
    print(f"🔧 Rate L2 reg: {r2_reg}")
    print(f"📊 Model index: {index}")
    print(f"💻 Device: {'CUDA' if is_cuda else 'CPU'}")
    
    if use_piezo:
        print(f"🫀 Piezo interface: ✅ ENABLED")
        print(f"   - Two-phase training (cardiac pretraining → main task)")
        print(f"   - Cardiac modulation during cognitive task")
        print(f"   - Enhanced network connectivity")
    else:
        print(f"🫀 Piezo interface: ❌ DISABLED")
        print(f"   - Standard Bi & Zhou architecture")
        print(f"   - Original training protocol")
    
    print("=" * 60)

    # Confirm before starting (for interactive sessions)
    if os.isatty(sys.stdin.fileno()):  # Only prompt if running interactively
        try:
            confirmation = input("🤔 Start training with these settings? [y/N]: ").lower()
            if confirmation not in ['y', 'yes']:
                print("❌ Training cancelled by user")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n❌ Training cancelled by user")
            sys.exit(0)

    print("\n🚀 Starting training...")

    # Start training with the specified parameters
    try:
        train_model(rule_name, w2_reg, r2_reg, index, use_piezo=use_piezo, is_cuda=is_cuda)
        
        # Success message
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        if use_piezo:
            print("🫀 Piezo-enhanced model training finished")
            print("   - Cardiac pretraining completed")
            print("   - Main task training with piezo modulation completed")
        else:
            print("🧠 Standard model training finished")
        print(f"💾 Model saved with index: {index}")
        print("✅ Ready for analysis and inference")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
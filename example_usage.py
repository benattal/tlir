"""
Example usage of the modular radiance field reconstruction library.

This script demonstrates how to use the refactored components to run
radiance field reconstruction experiments.
"""

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

# Import our modular components
from integrators import RadianceFieldPRB, RadianceFieldPRBRT
from training import (
    TrainingConfig, 
    train_radiance_field, 
    create_sensors, 
    create_scene,
    copy_parameters
)
from visualization import (
    plot_list, 
    plot_comparison, 
    plot_loss_curves,
    TrainingLogger
)
from config import (
    create_default_config, 
    create_ratio_tracking_config,
    setup_environment,
    validate_config
)

# Set up Mitsuba
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_basic_experiment():
    """Run a basic radiance field reconstruction experiment."""
    print("Running basic radiance field experiment...")
    
    # Create configuration
    config = create_default_config("basic_experiment")
    config.num_stages = 3
    config.num_iterations_per_stage = 10
    config.render_res = 128
    config.sensor_count = 5
    
    # Validate configuration
    issues = validate_config(config)
    if issues:
        print("Configuration issues:", issues)
        return
    
    # Set up environment
    setup_environment(config)
    
    # Create sensors
    sensors = create_sensors(
        num_sensors=config.sensor_count,
        render_res=config.render_res,
        center=config.camera_center,
        radius=config.camera_radius
    )
    
    # Load reference scene and render reference images
    scene_ref = mi.load_file(config.scene_file)
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=64) for i in range(config.sensor_count)]
    
    # Create training scene
    scene = create_scene(config.integrator_type)
    
    # Set up logger
    logger = TrainingLogger(os.path.join(config.output_dir, "training.log"))
    
    # Train the model
    results = train_radiance_field(
        scene=scene,
        sensors=sensors,
        ref_images=ref_images,
        config=config,
        progress_callback=logger.log_progress
    )
    
    # Log completion
    logger.log_training_complete(results['losses'])
    
    # Render final results
    final_images = [mi.render(scene, sensor=sensors[i], spp=128) for i in range(config.sensor_count)]
    
    # Visualize results
    plot_list(ref_images, 'Reference Images')
    plot_list(final_images, 'Final Results')
    
    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(results['losses'])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results


def run_comparison_experiment():
    """Run a comparison between regular and ratio tracking integrators."""
    print("Running integrator comparison experiment...")
    
    # Create configurations for both methods
    config_prb = create_default_config("prb_comparison")
    config_prb.num_stages = 2
    config_prb.num_iterations_per_stage = 15
    config_prb.render_res = 128
    config_prb.sensor_count = 5
    
    config_rt = create_ratio_tracking_config("rt_comparison")
    config_rt.num_stages = 2
    config_rt.num_iterations_per_stage = 15
    config_rt.render_res = 128
    config_rt.sensor_count = 5
    
    # Set up environment
    setup_environment(config_prb)
    
    # Create sensors
    sensors = create_sensors(
        num_sensors=config_prb.sensor_count,
        render_res=config_prb.render_res
    )
    
    # Load reference images
    scene_ref = mi.load_file(config_prb.scene_file)
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=64) for i in range(config_prb.sensor_count)]
    
    # Train with regular integrator
    print("Training with RadianceFieldPRB...")
    scene_prb = create_scene('rf_prb')
    results_prb = train_radiance_field(scene_prb, sensors, ref_images, config_prb)
    
    # Train with ratio tracking integrator
    print("Training with RadianceFieldPRBRT...")
    scene_rt = create_scene('rf_prb_rt')
    results_rt = train_radiance_field(scene_rt, sensors, ref_images, config_rt)
    
    # Render final results
    final_images_prb = [mi.render(scene_prb, sensor=sensors[i], spp=128) for i in range(config_prb.sensor_count)]
    final_images_rt = [mi.render(scene_rt, sensor=sensors[i], spp=128) for i in range(config_rt.sensor_count)]
    
    # Compare results
    plot_comparison(final_images_prb, final_images_rt, 
                   "RadianceFieldPRB", "RadianceFieldPRBRT")
    
    # Plot loss comparison
    plot_loss_curves({
        'RadianceFieldPRB': results_prb['losses'],
        'RadianceFieldPRBRT': results_rt['losses']
    })
    
    return results_prb, results_rt


def run_quick_test():
    """Run a quick test with minimal parameters."""
    print("Running quick test...")
    
    # Create quick configuration
    config = create_quick_config("quick_test", num_stages=1, num_iterations_per_stage=3)
    
    # Set up environment
    setup_environment(config)
    
    # Create sensors
    sensors = create_sensors(
        num_sensors=config.sensor_count,
        render_res=config.render_res
    )
    
    # Load reference images
    scene_ref = mi.load_file(config.scene_file)
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=32) for i in range(config.sensor_count)]
    
    # Create and train scene
    scene = create_scene(config.integrator_type)
    results = train_radiance_field(scene, sensors, ref_images, config)
    
    # Quick visualization
    final_images = [mi.render(scene, sensor=sensors[i], spp=64) for i in range(config.sensor_count)]
    plot_list(final_images, 'Quick Test Results')
    
    return results


if __name__ == "__main__":
    # Run different experiments based on command line argument
    import sys
    
    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = "quick"
    
    if experiment_type == "basic":
        run_basic_experiment()
    elif experiment_type == "comparison":
        run_comparison_experiment()
    elif experiment_type == "quick":
        run_quick_test()
    else:
        print("Available experiments: basic, comparison, quick")
        print("Usage: python example_usage.py [experiment_type]")

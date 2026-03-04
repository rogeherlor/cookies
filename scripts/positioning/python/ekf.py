import os
import logging
import json
import numpy as np
import pymap3d as pm
from pathlib import Path
from datetime import datetime
import metrics
import data_loader
import visualize
import visualize_state
import ekf_core
import ekf_config

################### LOAD DATA ###########################
nav_data        = ekf_config.NAV_DATA
t1              = ekf_config.OUTAGE_START
d               = ekf_config.OUTAGE_DURATION
use_3d_rotation = ekf_config.USE_3D_ROTATION

# Parameters castellana: Qpos=5.312e+00, Qvel=4.702e-02, Rpos=67.79, beta_acc=-1.910e-06, beta_gyr=-7.077e-02
# 2026-03-04 10:34:51,966 - INFO -   Initial P: pos_std=0.23m, vel_std=0.17m/s, orient_std=0.239rad

# Extract data arrays
accel_flu = nav_data.accel_flu
gyro_flu = nav_data.gyro_flu
vel_enu = nav_data.vel_enu
lla = nav_data.lla
orient = nav_data.orient

########################## VAR INIT ###############
frecIMU = nav_data.sample_rate
lla0    = nav_data.lla0

t2 = t1 + d
A  = int(t1 * frecIMU)
B  = int(t2 * frecIMU)

# EKF Parameters — from ekf_config (None → falls back to DEFAULT_EKF_PARAMS)
ekf_params = ekf_config.EKF_PARAMS


################## RUN EKF #########################
outage_config = {'start': t1, 'duration': d}
ekf_result = ekf_core.run_ekf(nav_data, ekf_params, outage_config, use_3d_rotation)

# Extract results
p = ekf_result['p']
v = ekf_result['v']
r = ekf_result['r']
bias_acc = ekf_result['bias_acc']
bias_gyr = ekf_result['bias_gyr']
std_pos = ekf_result['std_pos']
std_vel = ekf_result['std_vel']
std_orient = ekf_result['std_orient']
std_bias_acc = ekf_result['std_bias_acc']
std_bias_gyr = ekf_result['std_bias_gyr']

########################### RESULTS AND VISUALISATION ###########################

# Convert ground truth trajectory to ENU array format
f = pm.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],lla0[0],lla0[1],lla0[2])
f_array = np.column_stack([f[0], f[1], f[2]])  # Nx3 array [E, N, U]

# Prepare GNSS outage information
gnss_outage_info = {
    'start': t1,
    'end': t2,
    'duration': d,
    'start_idx': A,
    'end_idx': B
}

# Run comprehensive evaluation
results = metrics.evaluate_navigation_performance(
    p_est=p,
    v_est=v,
    r_est=r,
    p_gt=f_array,
    v_gt=vel_enu,
    r_gt=orient,
    dataset_name=nav_data.dataset_name,
    gnss_outage_info=gnss_outage_info,
    sample_rate=frecIMU
)

# Create output directories
base_dir = Path(__file__).parent
logs_dir = os.path.join(base_dir, '../../../logs')
base_outputs_dir = os.path.join(base_dir, '../../../outputs/ekf')

# Generate run ID based on dataset and GNSS outage configuration
if t1 == 0 and d == 0:
    run_id = "no_outage"
    trajectory_folder = f"{nav_data.dataset_name}_no_outage"
else:
    run_id = f"outage_{t1}s_{d}s"
    trajectory_folder = f"{nav_data.dataset_name}_outage_{t1}s_{d}s"

# Create trajectory-specific output directory
outputs_dir = os.path.join(base_outputs_dir, trajectory_folder)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'ekf_errors_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log all evaluation results
metrics.log_evaluation_results(logger, results, log_file)

# Save numerical results to outputs
results_file = os.path.join(outputs_dir, f'{run_id}_results.json')
# Convert numpy arrays to lists for JSON serialization
results_serializable = {
    'dataset': results['dataset'],
    'total_samples': results['total_samples'],
    'gnss_outage': results['gnss_outage'],
    'position_rmse': results['position_rmse'],
    'velocity_rmse': results['velocity_rmse'],
    'orientation_rmse': results['orientation_rmse'],
    'ate': {k: v for k, v in results['ate'].items() if k != 'errors'},
    'rte_1s': {k: v for k, v in results['rte_1s'].items() if k != 'errors'},
    'rte_5s': {k: v for k, v in results['rte_5s'].items() if k != 'errors'},
    'rte_10s': {k: v for k, v in results['rte_10s'].items() if k != 'errors'},
    'peak_errors': results['peak_errors'],
    'outage_analysis': results['outage_analysis']
}
with open(results_file, 'w') as f:
    json.dump(results_serializable, f, indent=2)
logger.info(f'Results saved to: {results_file}')

# Save trajectory data
np.savez(os.path.join(outputs_dir, f'{run_id}_trajectories.npz'),
         p_est=p, v_est=v, r_est=r,
         p_gt=f_array, v_gt=vel_enu, r_gt=orient,
         bias_acc=bias_acc, bias_gyr=bias_gyr,
         std_pos=std_pos, std_vel=std_vel, std_orient=std_orient,
         std_bias_acc=std_bias_acc, std_bias_gyr=std_bias_gyr,
         time=np.arange(len(p)) / frecIMU)

########################### GENERATE CHARTS ###########################

# Generate all visualizations using visualize module
generated_files = visualize.generate_all_plots(
    results=results,
    p_est=p,
    v_est=v,
    r_est=r,
    p_gt=f_array,
    v_gt=vel_enu,
    r_gt=orient,
    sample_rate=frecIMU,
    output_dir=outputs_dir,
    run_id=run_id,
    accel_flu=accel_flu,
    gyro_flu=gyro_flu,
    lla0=lla0,
    gps_available=nav_data.gps_available
)

logger.info(f'Generated {len(generated_files)} charts in: {outputs_dir}/')

# Generate EKF state visualizations
time_array = np.arange(len(p)) / frecIMU

logger.info('Generating EKF state visualizations...')

visualize_state.plot_bias_estimates(
    time_array, bias_acc, bias_gyr, gnss_outage_info,
    save_path=os.path.join(outputs_dir, f'{run_id}_bias_estimates.png')
)
generated_files.append(f'{run_id}_bias_estimates.png')

visualize_state.plot_uncertainty_evolution(
    time_array, std_pos, std_vel, std_orient, gnss_outage_info,
    save_path=os.path.join(outputs_dir, f'{run_id}_uncertainty.png')
)
generated_files.append(f'{run_id}_uncertainty.png')

visualize_state.plot_bias_uncertainty(
    time_array, bias_acc, bias_gyr, std_bias_acc, std_bias_gyr, gnss_outage_info,
    save_path=os.path.join(outputs_dir, f'{run_id}_bias_uncertainty.png')
)
generated_files.append(f'{run_id}_bias_uncertainty.png')

# Filter consistency check
errors_pos = f_array - p
visualize_state.plot_filter_consistency(
    time_array, errors_pos, std_pos, gnss_outage_info,
    save_path=os.path.join(outputs_dir, f'{run_id}_filter_consistency.png')
)
generated_files.append(f'{run_id}_filter_consistency.png')

logger.info(f'Total generated charts: {len(generated_files)}')
logger.info(f'Run ID: {run_id}')

# Display interactive trajectory plot
visualize.show_interactive_plot(p, f_array, gnss_outage_info, nav_data.dataset_name, lla0=lla0, gps_available=nav_data.gps_available)

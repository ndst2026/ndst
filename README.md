# NDST: Neural Driving Style Transfer

This repository provides an implementation of **Neural Driving Style Transfer (NDST)**, a vision-based autonomous driving personalization framework. NDST transfers a driver's individual driving style to an autonomous vehicle model by adding a Personalized Block (PB) on top of a pre-trained Baseline Driving Model (BDM).

The system is designed for ROS/Gazebo-based autonomous driving simulation and supports both offline training and online inference.

---


## Repository Structure

```text
ndst/
├── catkin_ws/                  # ROS catkin workspace
│   └── src/
│       ├── fusion/             # Gazebo/ROS vehicle simulation package
│       └── run_ndst/           # ROS node for NDST inference
│           └── src/
│               └── run_neural.py
│
├── code/                       # Neural network training and utility code
│   ├── config.py               # Loads YAML configuration files
│   ├── const.py                # Constant definitions
│   ├── dataset_utils.py        # Dataset, image processing, runtime utilities
│   ├── model.py                # NDST model architecture
│   ├── train.py                # Training script
│   └── requirements.txt        # Python dependencies
│
├── config/                     # Configuration files
│   ├── ndst_config.yaml
│   ├── train_ndst/
│   │   └── train_ndst.yaml
│   └── run_ndst/
│       └── run_ndst.yaml
│
├── model/                      # Model directory
│   ├── base_model/             # Baseline Driving Model weights
│   ├── style_a_model/          # Personalized model for Style A
│   └── style_b_model/          # Personalized model for Style B
│
├── setup.bash                  # Environment setup script
├── start_fusion.sh             # Gazebo simulation launch script
├── LICENSE
└── README.md
```

---

## Requirements

Recommended environment:

```text
Ubuntu 18.04
ROS Melodic
Python 2.7
TensorFlow 1.x
Keras 2.x
OpenCV
Gazebo
```

## Installation

Clone this repository:

```bash
cd ndst
```

Create Python environment:

```bash
conda create -n ndst python=2.7 -y
conda activate ndst
```

Install Python dependencies:

```bash
pip install -r code/requirements.txt
```

Build ROS workspace:

```bash
cd catkin_ws
catkin_make
cd ..
```

The `setup.bash` script configures the main runtime paths:

```bash
source setup.bash
```

It sets the following paths:

```text
NDST_PATH
GAZEBO_MODEL_PATH
PYTHONPATH
catkin_ws/devel/setup.bash
```

After sourcing the setup file, check whether the Python modules are correctly loaded:

```bash
python -c "import dataset_utils; print(dataset_utils.__file__)"
```

The output should point to:

```text
/path/to/ndst/code/dataset_utils.py
```

---


## Dataset Format

The dataset is organized by split and driving style. Each driving sequence is stored in a separate folder. The folder contains image files and one CSV file, and the CSV file name must match the folder name.

Example:

```text
dataset/
├── train/
│   ├── style_a/
│   │   └── 09-12-14-37-22/
│   │       ├── 09-12-14-37-22.csv
│   │       ├── 04-19-16-04-571482.jpg
│   │       └── ...
│   └── style_b/
│       └── 09-02-20-05-44/
│           ├── 09-02-20-05-44.csv
│           ├── 04-19-16-04-571482.jpg
│           └── ...
└── test/
    ├── style_a/
    └── style_b/
```

Each CSV file should contain the following columns:
```text
image_fname,
steering_angle, throttle, brake,
linux_time,
vel, vel_x, vel_y, vel_z,
accel_x, accel_y,
pos_x, pos_y, pos_z,
goal_vel
```

Column description:
```text
image_fname      Image filename corresponding to the current row
steering_angle   Steering command
throttle         Throttle command
brake            Brake command
linux_time       Timestamp
vel              Vehicle speed
vel_x            Vehicle velocity along the x-axis
vel_y            Vehicle velocity along the y-axis
vel_z            Vehicle velocity along the z-axis
accel_x          Vehicle acceleration along the x-axis
accel_y          Vehicle acceleration along the y-axis
pos_x            Vehicle position along the x-axis
pos_y            Vehicle position along the y-axis
pos_z            Vehicle position along the z-axis
goal_vel         Target velocity
```

---

## Training

The training script is located at:

```text
code/train.py
```

Basic usage:

```bash
cd /path/to/ndst
. setup.bash
python code/train.py <data_path> <base_model_path>
```

Example:

```bash
python code/train.py dataset/train/style_a/09-12-14-37-22/ model/base_model/base
```

```text
The dataset path should point to the driving-style dataset folder.
The CSV file and image files should follow the dataset format described above.
The model path should be given without .json or .h5.
Training options are controlled by config/train_ndst/train_ndst.yaml.
```

## Running the Gazebo Simulation

Use `start_fusion.sh` to build the catkin workspace and launch the simulation.

```bash
cd /path/to/ndst
. setup.bash
./start_fusion.sh track_style_train
```

Available world names include:

```text
track_style_train
track_style_test
```

Example:

```bash
./start_fusion.sh track_style_test
```

If the script is not executable, run:

```bash
chmod +x start_fusion.sh
```

---

## Running NDST Online Inference

The ROS inference node is located at:

```text
catkin_ws/src/run_ndst/src/run_neural.py
```

Basic usage:

```bash
cd /path/to/ndst
. setup.bash
rosrun run_ndst run_neural.py <style_model_path> <base_model_path>
```
The target velocity must be published separately through the /goal_velocity topic.

### Example:

Terminal 1: start the simulation.
```bash
cd /path/to/ndst
. setup.bash
rosrun run_ndst run_neural.py model/style_a_model/style_a model/base_model/base
```

Terminal 2: publish the target velocity.
```bash
cd /path/to/ndst
. setup.bash
rostopic pub /goal_velocity std_msgs/Float64 "data: 20.0"
```

Terminal 3: run the NDST inference node.
```bash
cd /path/to/ndst
. setup.bash
rosrun run_ndst run_neural.py model/style_a_model/style_a model/base_model/base
```


Arguments:

```text
<style_model_path> Path to the trained personalized NDST model without file extension
<base_model_path>  Path to the pre-trained baseline model without file extension
```

For example, if the model files are:

```text
model/style_a_model/style_a.h5
model/base_model/base.h5
```

then the command should use:

```text
../../model/style_a_model/style_a
../../model/base_model/base
```


### Style-specific scaling parameters

Before running inference, set the scaling parameters in config/train_ndst/train_ndst.yaml to match the style model.
These parameters should be the same as those used during training.

For style A:
```text
steering_angle_scale: 10.0
throttle_scale: 5.0
brake_scale: 5.0
```

For style B:
```text
steering_angle_scale: 5.0
throttle_scale: 1.0
brake_scale: 1.0
```

The scaling values are style-dependent because each driving dataset has a different control range. For example, if the throttle range is approximately -0.2 to 0.2, throttle_scale is set to 5.0 so that the scaled target range becomes approximately -1.0 to 1.0.

---

## Notes

- The code was originally designed for ROS/Gazebo-based autonomous driving simulation.
- Paths in the examples should be adjusted to the local environment.
- The trained model weight files may need to be downloaded separately.
- When running in ROS Melodic, ensure that the Python version and TensorFlow/Keras versions are compatible.

---

## Citation

If this repository is used for academic work, please cite the corresponding paper or project documentation.

```bibtex
@misc{ndst,
  title = {Neural Driving Style Transfer for Human-Like Vision-Based Autonomous Driving},
  author = {Anonymous},
  year = {2026},
  note = {Code repository}
}
```


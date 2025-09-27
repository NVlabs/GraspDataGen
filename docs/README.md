
# GraspDataGen Documentation

Welcome to the GraspDataGen documentation! This project is designed to create collision-free, geometrically plausible grasps for triangle mesh objects and USD grippers, then validate them through physics simulation.

## Overview

### Components

GraspDataGen is a standalone data generation tool but it can also be used to generate data for training new [GraspGen](https://github.com/NVlabs/GraspGen) models. It consists of three main components:

1. **[Gripper Definition](components/gripper-definition.md)** - Reads USD gripper files and extracts parameters needed for grasp generation
2. **[Grasp Guess Generation](components/grasp-guess.md)** - Creates geometrically plausible grasps for objects
3. **[Grasp Verification with Simulation](components/grasp-sim.md)** - Tests grasp validity using PhysX simulation

### Example Workflows

The components are meant to be used individually or as building blocks to a complete grasp generation pipeline. The following are some examples and workflows that can be built with the components:

1. **[Complete Data Generation Pipeline](workflows/datagen.md)** - Generate and verify grasps for a list of objects and scales
2. **[Generate Data for Grasp Gen](workflows/graspgen.md)** - Generate grasp data specifically for [Grasp Gen](https://github.com/NVlabs/GraspGen)
3. **[Gripper Setup](examples/gripper-setup.md)** - Create a gripper definition and check it visually
4. **[Running the Gripper Definition Component](examples/gripper-definition.md)** - Visually verify your gripper will work with the simulation
5. **[Single Object Grasp Guess Generation](examples/grasp-guess.md)** - Create collision-free grasps for a single object
6. **[Verify Grasps with Simulation](examples/grasp-sim.md)** - Verify user defined grasps with simulation

### Gripper Configurations, Default Parameters and Overrides

GraspDataGen has default parameters for all its components, and all of those defaults can be overridden with command line arguments or grouped together with gripper configurations. It's worth understanding GraspDataGen's **[parameter and configuration system](api/parameter-system.md)**.

## Quick Start

### Installation

For detailed installation instructions, see the **[Installation Guide](installation.md)**. The guide covers Docker installation (recommended), integration with existing IsaacLab installations, and troubleshooting.

## Documentation Structure

### Getting Started

- **[Installation Guide](installation.md)** - Complete installation instructions for all environments

### Settings and Parameter System

- **[Parameters and Overrides](api/parameter-system.md)** - How configurations, parameters and overrides work

### Core Components

- **[Gripper Definition](components/gripper-definition.md)** - Creating and configuring gripper models
- **[Grasp Guess Generation](components/grasp-guess.md)** - Generating geometrically plausible grasps
- **[Grasp Simulation](components/grasp-sim.md)** - Physics-based grasp validation

### Workflows

- **[Complete Data Generation Pipeline](workflows/datagen.md)** - Generate and verify grasps for a list of objects and scales
- **[Generate Data for Grasp Gen](workflows/graspgen.md)** - Generate grasp data specifically for [Grasp Gen](https://github.com/NVlabs/GraspGen)

### Tools

- **[Tools Overview](tools/README.md)** - Overview of available utility tools
- **[Compare Tools](tools/compare-tools.md)** - Tools for comparing grasp simulations and gripper configurations
- **[Utility Tools](tools/utility-tools.md)** - General utility scripts for data processing

### Examples

- **[Gripper Setup](examples/gripper-setup.md)** - Create a gripper definition and check it visually
- **[Running the Gripper Definition Component](examples/gripper-definition.md)** - Visually verify your gripper will work with the simulation
- **[Single Object Grasp Guess Generation](examples/grasp-guess.md)** - Create collision-free grasps for a single object
- **[Verify Grasps with Simulation](examples/grasp-sim.md)** - Verify user-defined grasps with simulation

## Architecture

The system is designed to be modular, with each component able to run independently or as part of a larger pipeline:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gripper       │    │   Grasp Guess    │    │   Grasp Sim     │
│   Definition    │───▶│   Generation     │───▶│   Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   USD Gripper File      Grasp Guess Data      Validated Grasps
```


<sourcefile>"""""
Goal : Coding an application that assisted with AI can function as an experienced electrician tasked with estimating the labor time for drilling through framing members to run electrical wiring in a residential home, by analyzing a photo, and following the request


_______________________________________________________

this will be done in multiple steps

step 1 Guide for coding
create logic behind the idea, how ai factors in, math behind the complexities

step 2
coding- broken down into sections building out the coding for the ai application, provide a prompt for each section or file, to be done a slow enough pace to to withstand the complexities at hand.

step 3 - clean up, and test,

step 4 Enhance

Step 5 produce v1 (desktop application)
_______________________________________________________

We are on STEP 1

step 1 Guide for coding
create logic behind the idea, how ai factors in, math behind the complexities
_______________________________________________________


Examples of things to consider, but do not limit your ideas to.. Everything under this , is only an exmaple of the thought proccess, use your own thoughts to enhance upon that to begin step 1 of this goal


Framing
This indicates which part of the house the wire run is in (e.g.,
Basement, 1st Floor, 2nd Floor, Attic).



This specifies whether you're running through walls (studs),
ceilings/floors (joists), or roof (rafters/trusses).

Distance the wire is ran
This is the linear distance of the wire run within the framed section, measured in feet.

framing space
This is the spacing of the framing members in that location, measured in inches.

Complexeties
This describes various factors that may affect the complexity of the job, such as wood thickness/type, wire type/thickness, working space, obstructions, and drilling technique.

Follow these steps to estimate the labor time:

1. Calculate the number of framing members to drill through:
a. Convert the framing spacing from inches to feet.
b. Use the formula: Number of Members = CEILING(Wire Run Distance / Spacing in feet)
c. Round up to the nearest whole number.

2. Determine the complexity level based on the provided complexity factors:
- Simple: thin wire, easy drilling, clear space
- Moderate: standard wire, some obstructions, typical drilling
- Complex: thick wire, difficult drilling, many obstructions, tight space, thicker framing

3. Estimate the time per member based on the complexity level:
- Simple: 5-10 minutes per member
- Moderate: 10-15 minutes per member
- Complex: 15-20+ minutes per member

4. Calculate the total estimated time:
Total Estimated Time = Number of Members * Time per Member

Provide your answer in the following format:

Number of framing members to drill through: [number]
Complexity level: [Simple/Moderate/Complex]
Estimated time per member: [range in minutes]
Total estimated time: [range in minutes or hours]

Explanation: [Briefly explain your reasoning for the complexity level and time estimates, referencing the provided information]


# Step 1: Guide for Coding an AI-Assisted Electrician Time Estimation Application

## System Architecture Overview

The application will use computer vision and machine learning to analyze photos of residential framing members and estimate the labor time required for running electrical wiring. Here's the logical framework:

### Core Components:

1. **Photo Analysis Module**
- Processes uploaded images using computer vision
- Identifies framing members (studs, joists, rafters)
- Detects obstacles and existing utilities
- Measures distances and spacing

2. **Complexity Assessment Engine**
- Evaluates material types (wood species, engineered lumber, metal)
- Assesses space constraints and accessibility
- Identifies drilling challenges
- Calculates overall complexity score

3. **Time Estimation Algorithm**
- Calculates number of penetrations required
- Applies time factors based on complexity
- Generates time ranges with confidence levels

4. **User Interface**
- Photo upload capability
- Parameter input/confirmation
- Results display with detailed breakdown
- Saving/exporting functionality

## AI Integration

### Computer Vision Capabilities:
- **Object Detection**: Identify framing members, obstacles, and utilities
- **Semantic Segmentation**: Classify materials and spaces
- **Measurement Estimation**: Calculate distances and dimensions

### Model Selection:
- Primary model: YOLOv8 or Faster R-CNN for framing member detection
- Supporting models:
- Depth estimation network for 3D space understanding
- Material classification model for wood type identification
- Instance segmentation for obstacle detection

### Training Requirements:
- Labeled dataset of residential construction photos
- Augmentations for various lighting conditions
- Transfer learning from pretrained construction/building models

## Mathematical Framework

### 1. Framing Member Count Calculation

Number of Members = CEILING(Wire Run Distance / Spacing in feet)

Where:
- Wire Run Distance: Length of planned wire path (feet)
- Spacing in feet = Framing Spacing (inches) / 12

### 2. Complexity Scoring System
Start with base score of 1.0, then add factors:

**Material Factors:**
- Softwood (pine, spruce): +0.0
- Hardwood (oak, maple): +0.3
- Engineered lumber: +0.2
- Metal framing: +0.5

**Wire Factors:**
- Standard 14/12 AWG: +0.0
- 10 AWG or larger: +0.2
- Multiple wires: +0.2 per additional wire

**Workspace Factors:**
- Open accessible space: +0.0
- Confined space: +0.3
- Overhead work: +0.2
- Poor lighting: +0.2

**Obstruction Factors:**
- Clean framing: +0.0
- Existing utilities: +0.2 per type
- Insulation present: +0.2
- Structural elements to avoid: +0.3

**Final Complexity Level:**
- Simple: Score < 1.5
- Moderate: Score 1.5-2.5
- Complex: Score > 2.5

### 3. Time Estimation Formula

Per-Member Time = Base Time × Complexity Multiplier
Total Time = Number of Members × Per-Member Time


Where:
- Base Time:
- Simple: 7 minutes
- Moderate: 12 minutes
- Complex: 17 minutes
- Complexity Multiplier: Adjusts based on specific factors

### 4. Confidence Calculation

Confidence Score = (AI Detection Confidence × 0.6) + (Input Data Completeness × 0.4)

- Provide time ranges that widen as confidence decreases

## Technical Implementation Approach

### Backend Technology Stack:
- Python for AI/ML processing
- TensorFlow or PyTorch for computer vision models
- FastAPI for API development

### Frontend Options:
- Desktop: Electron for cross-platform compatibility
- Web application: React for responsive interface
- Mobile: React Native for on-site usage

### Development Phases:
1. Core AI model development and training
2. Calculation engine implementation
3. API layer for model interaction
4. User interface development
5. Testing and validation
6. Deployment and packaging

## Key Challenges and Solutions

1. **Variable Photo Quality**
- Implement image enhancement preprocessing
- Provide guidance for optimal photo angles
- Allow manual correction of AI detections

2. **Complex 3D Spaces**
- Request multiple photos from different angles
- Use depth estimation to understand spatial relationships
- Implement 3D reconstruction for complex layouts

3. **Regional Building Differences**
- Include region-specific framing databases
- Allow for custom parameter adjustments
- Train multiple regional models

4. **Confidence Handling**
- Clearly indicate low-confidence estimates
- Request additional information when needed
- Provide explanation of estimation factors

This framework provides the foundation for building an advanced AI-assisted application that can accurately estimate electrical wiring labor time based on construction photo analysis.




# AI-Assisted Electrician Time Estimation Application: Development Prompts

## Project Setup Prompts

### Prompt 1: Environment and Project Structure Setup

Create a Python project structure for an AI-assisted electrician time estimation application. The application should:
1. Use a virtual environment with Python 3.9+
2. Include directories for models, data, utils, api, and frontend
3. Set up a requirements.txt file with essential packages (TensorFlow/PyTorch, OpenCV, FastAPI, etc.)
4. Configure basic logging and error handling
5. Implement a modular architecture that separates concerns

Provide the complete directory structure and configuration files needed to begin development.


### Prompt 2: Data Collection and Preparation Framework

Develop a data handling module for the electrician time estimation application that can:
1. Process and store training images of residential framing members
2. Implement data augmentation for various lighting conditions and angles
3. Create annotation tools/scripts for labeling framing members, obstacles, and materials
4. Build a data pipeline that converts raw images to model-ready formats
5. Include functions to split data into training/validation/test sets

The module should handle common image formats and include proper error handling for corrupted files.


## AI Model Development Prompts

### Prompt 3: Framing Member Detection Model

Implement a computer vision model to detect and classify residential framing members from images. The model should:
1. Identify studs, joists, rafters, and other structural elements
2. Use a YOLOv8 or Faster R-CNN architecture with pretrained weights
3. Include training code with appropriate hyperparameters
4. Implement evaluation metrics (mAP, precision, recall)
5. Save model checkpoints and export to an inference-optimized format

Provide complete model definition, training loop, and inference functions with documentation.


### Prompt 4: Distance and Measurement Estimation

Create a module that estimates measurements from detected framing members in images. The module should:
1. Calculate the spacing between framing members
2. Estimate the dimensions of framing members (2x4, 2x6, etc.)
3. Implement a reference scale detection system or manual calibration
4. Calculate the total run distance for wiring paths
5. Provide confidence scores for measurements

Include visualization functions to display detected measurements on images for user verification.


### Prompt 5: Complexity Assessment System

Develop an algorithmic system that assesses the complexity of electrical wiring installation based on detected elements. The system should:
1. Analyze detected materials and assign appropriate difficulty scores
2. Identify and score obstacles and space constraints
3. Evaluate access difficulties from image context
4. Implement the complete complexity scoring formula from the specifications
5. Classify jobs into Simple, Moderate, and Complex categories

Provide the scoring system implementation with clear documentation of each factor's contribution.
\
## Backend Processing Prompts

### Prompt 6: Time Estimation Core Algorithm

Implement the core time estimation algorithm that calculates labor time based on detected elements and complexity scores. The algorithm should:
1. Calculate the number of framing members to drill through
2. Apply the appropriate time factors based on complexity
3. Generate time ranges with confidence intervals
4. Account for special conditions (overhead work, confined spaces)
5. Provide detailed breakdowns of time components

Include comprehensive unit tests to verify accuracy across various scenarios.


### Prompt 7: API Development

Create a RESTful API using FastAPI that serves the electrical time estimation functionality. The API should:
1. Accept image uploads and additional parameters
2. Process images through the detection and estimation pipeline
3. Return structured JSON responses with time estimates and confidence scores
4. Implement proper error handling and validation
5. Include authentication for production use
6. Provide API documentation using Swagger/OpenAPI

Ensure the API is properly structured with models, routes, and dependency injection.


## Frontend Development Prompts

### Prompt 8: Desktop Application UI (Electron)

Develop an Electron-based desktop application frontend for the electrician time estimation tool. The UI should:
1. Provide an intuitive image upload interface with drag-and-drop
2. Display detected framing members with visual overlays
3. Allow users to adjust detected elements if needed
4. Present time estimates with clear breakdowns and explanations
5. Enable saving/exporting of results in PDF and CSV formats
6. Implement responsive design for various screen sizes

Include all necessary HTML, CSS, and JavaScript files with proper documentation.


### Prompt 9: Parameter Input and Adjustment Interface

Create the user interface components for manual parameter input and adjustment. These should:
1. Allow users to specify job details not detectable from images
2. Provide forms for adjusting detected measurements if needed
3. Include material selection dropdowns with common options
4. Implement sliders for complexity factor adjustment
5. Show real-time updates to time estimates when parameters change

Ensure all inputs have appropriate validation and user feedback.


## Integration and Testing Prompts

### Prompt 10: End-to-End Integration

Implement code that integrates all components of the electrician time estimation application. This should:
1. Connect the frontend UI to the backend API
2. Ensure proper data flow between detection, estimation, and display components
3. Implement error handling and recovery throughout the pipeline
4. Add progress indicators for long-running processes
5. Create a configuration system for application settings

Include comprehensive integration tests that verify the complete workflow.


### Prompt 11: Testing Framework and Validation

Develop a comprehensive testing framework for the application that includes:
1. Unit tests for individual components and algorithms
2. Integration tests for component interactions
3. End-to-end tests simulating real user workflows
4. Performance benchmarking for processing times
5. Accuracy validation against known examples with ground truth
6. Test fixtures and mock data generation utilities

Ensure test coverage across all critical components with automated reporting.


## Deployment and Enhancement Prompts

### Prompt 12: Application Packaging and Deployment

Create the necessary scripts and configuration to package the application for distribution. This should:
1. Bundle the Electron application with all dependencies
2. Configure proper versioning and update mechanisms
3. Implement installer generation for Windows/Mac/Linux
4. Set up proper logging and error reporting for production use
5. Include documentation and help resources within the application

Provide complete deployment instructions and verification procedures.


### Prompt 13: Performance Optimization

Implement optimizations to enhance the performance and user experience of the application:
1. Model quantization techniques to speed up inference
2. Image preprocessing optimizations for faster detection
3. Caching mechanisms for repeated calculations
4. Background processing for time-intensive tasks
5. UI optimizations for responsiveness
6. Memory management improvements

Include before/after benchmarks demonstrating the performance gains.


### Prompt 14: Enhancement for Mobile Use

Extend the application to support mobile usage through:
1. A React Native implementation for iOS and Android
2. Camera integration for direct photo capture
3. Cloud synchronization of estimates and projects
4. Offline functionality for job sites without connectivity
5. Touch-optimized UI adjustments
6. Battery usage optimizations

Provide the additional code and modifications needed while maintaining the core estimation logic.


## Advanced Features Prompts

### Prompt 15: 3D Space Reconstruction

Implement advanced 3D reconstruction capabilities that:
1. Process multiple images to create a 3D understanding of the space
2. Generate a basic 3D model of the framing structure
3. Allow users to trace wiring paths through the 3D model
4. Calculate optimal routing to minimize drilling
5. Visualize the complete wiring path in 3D

Include algorithms for image registration, depth estimation, and 3D visualization.


### Prompt 16: Report Generation and Documentation

Create a comprehensive reporting module that:
1. Generates detailed PDF reports of time estimations
2. Includes annotated images highlighting detection results
3. Provides material lists and drilling specifications
4. Creates project documentation for clients and crew
5. Supports customizable templates and branding
6. Implements export to various formats (PDF, Excel, CSV)

Provide the complete reporting implementation with example outputs.


These structured prompts break down the development process into manageable sections, ensuring that each component is properly implemented before integration. Following this sequence will lead to a stable, well-architected application that meets the requirements.
"""""
""""
Prompt 1

# Electrician Time Estimator: Project Structure Setup

Below is a complete project structure and configuration for the AI-assisted electrician time estimation application.

## Directory Structure


electrician-estimator/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── run.py
├── config/
│ ├── __init__.py
│ ├── logging_config.py
│ └── settings.py
├── api/
│ ├── __init__.py
│ ├── main.py
│ ├── routes/
│ │ ├── __init__.py
│ │ └── estimation.py
│ ├── models/
│ │ ├── __init__.py
│ │ └── schemas.py
│ └── middleware/
│ ├── __init__.py
│ └── error_handling.py
├── data/
│ ├── __init__.py
│ ├── loaders/
│ │ ├── __init__.py
│ │ └── image_loader.py
│ ├── processors/
│ │ ├── __init__.py
│ │ └── image_processor.py
│ └── storage/
│ ├── images/
│ ├── processed/
│ └── annotations/
├── models/
│ ├── __init__.py
│ ├── detection/
│ │ ├── __init__.py
│ │ └── framing_detector.py
│ ├── estimation/
│ │ ├── __init__.py
│ │ └── time_estimator.py
│ └── weights/
├── utils/
│ ├── __init__.py
│ ├── exceptions.py
│ ├── logger.py
│ └── visualization.py
├── frontend/
│ └── electron/
│ ├── package.json
│ ├── main.js
│ └── src/
│ ├── index.html
│ ├── styles/
│ │ └── main.css
│ └── scripts/
│ └── main.js
└── tests/
├── __init__.py
├── conftest.py
├── test_api/
│ ├── __init__.py
│ └── test_estimation.py
├── test_models/
│ ├── __init__.py
│ └── test_detection.py
└── test_utils/
├── __init__.py
└── test_visualization.py
"""
</sourcefile>"""
## Configuration Files

### requirements.txt

# Core dependencies
python-dotenv>=0.19.0
pydantic>=1.9.0

# API
fastapi>=0.88.0
uvicorn>=0.20.0
python-multipart>=0.0.5
httpx>=0.23.0

# Machine Learning & Computer Vision
tensorflow>=2.11.0 # For CPU
# tensorflow-gpu>=2.11.0 # Uncomment for GPU support
# OR
# torch>=1.13.0
# torchvision>=0.14.0
opencv-python>=4.7.0
pillow>=9.4.0
numpy>=1.23.5
scikit-image>=0.19.3
scikit-learn>=1.2.0
matplotlib>=3.6.3

# Image processing
albumentations>=1.3.0

# Utilities
tqdm>=4.64.1
loguru>=0.6.0

# Testing
pytest>=7.2.1
pytest-cov>=4.0.0

# Development
black>=23.1.0
flake8>=6.0.0
isort>=5.12.0


### .env.example

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=True

# Model Settings
MODEL_TYPE=tensorflow # or pytorch
DETECTION_MODEL=yolov8
CONFIDENCE_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO


### .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Environment variables
.env

# Model weights
models/weights/*.h5
models/weights/*.pth
models/weights/*.pb
models/weights/*.onnx

# Data files
data/storage/images/*
data/storage/processed/*
data/storage/annotations/*
data/storage/images/.gitkeep
data/storage/processed/.gitkeep
data/storage/annotations/.gitkeep

# Log files
logs/
*.log

# Test coverage
.coverage
htmlcov/

# Node.js dependencies
node_modules/

## Core Configuration Files

### config/logging_config.py

import logging
import os
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

def configure_logging(name="electrician_estimator", level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        name (str): Logger name
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers if they don't exist yet
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(logs_dir / f"{name}.log")
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
    
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
    
        return logger
        
        # Load image if path is provided
        def load_and_transform_image(image, custom_transform, save_path=None):
            if isinstance(image, (str, Path)):
                img = load_image(image)
            else:
                img = image.copy()
        
            # Apply transformation
            augmented = custom_transform(image=img)['image']
        
            # Save if path is provided
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(augmented, save_path)
        
            return augmented
        
        def get_image_dimensions(first_img):
            height, width = first_img.shape[:2]

                # Determine shape based on normalization
                if normalize:
                    batch = np.zeros((len(image_paths), height, width, 3), dtype=np.float32)
                else:
                    batch = np.zeros((len(image_paths), height, width, 3), dtype=np.uint8)

                # Process each image
                for i, img_path in enumerate(image_paths):
                    try:
                        processed_img = self.image_processor.preprocess_image(
                            img_path,
                            normalize=normalize,
                            enhance_contrast=True,
                            denoise=True
                        )

                        batch[i] = processed_img

                    except Exception as e:
                        logger.error(f"Failed to prepare image {img_path}: {str(e)}")
                        # Fill with zeros for failed images
                        batch[i] = np.zeros((height, width, 3), dtype=batch.dtype)

                return batch

# Load model
try:
    if weights_path is not None:
        # Load custom trained weights
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ModelNotFoundError(f"Weights file not found: {weights_path}")

        logger.info(f"Loading custom weights from {weights_path}")
        self.model = YOLO(str(weights_path))

    elif pretrained:
        # Load pretrained weights
        logger.info(f"Loading pretrained YOLOv8{model_size}")
        self.model = YOLO(f"yolov8{model_size}.pt")

        # Update number of classes if needed
        if self.model.names != YOLO_CONFIG['class_names']:
            logger.info(f"Updating model for {len(CATEGORIES)} framing categories")
            self.model.names = YOLO_CONFIG['class_names']
    else:
        # Initialize with random weights
        logger.info(f"Initializing YOLOv8{model_size} with random weights")
        self.model = YOLO(f"yolov8{model_size}.yaml")

    # Move model to device
    self.model.to(self.device)

except Exception as e:
    raise ModelNotFoundError(f"Failed to load YOLOv8 model: {str(e)}")

def detect(
    self,
    image: Union[str, Path, np.ndarray],
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    return_original: bool = False
) -> Dict:
    """
    Detect framing members in an image.

    Args:
        image: Image file path or numpy array
        conf_threshold: Confidence threshold for detections (overrides default)
        iou_threshold: IoU threshold for NMS (overrides default)
        return_original: Whether to include the original image in the results

    Returns:
        Dict: Detection results with keys:
        - 'detections': List of detection dictionaries
        - 'image': Original image (if return_original=True)
        - 'inference_time': Time taken for inference
    """
    # Use specified thresholds or fall back to instance defaults
    conf = conf_threshold if conf_threshold is not None else self.conf_threshold
    iou = iou_threshold if iou_threshold is not None else self.iou_threshold

    try:
        # Track inference time
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        inference_time = time.time() - start_time

        # Process results
        detections = []

        # Extract results from the first image (or only image)
        result = results[0]

        # Convert boxes to the desired format
        if len(result.boxes) > 0:
            # Get boxes, classes, and confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            # Format detections
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                x1, y1, x2, y2 = box

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # Get class name
                class_name = result.names[cls]

                detection = {
                    'id': i,
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'category_id': int(cls),
                    'category_name': class_name,
                    'confidence': float(score)
                }

                detections.append(detection)

        # Prepare return dictionary
        results_dict = {
            'detections': detections,
            'inference_time': inference_time
        }

        # Include original image if requested
        if return_original:
            if isinstance(image, (str, Path)):
                # If image is a path, get the processed image from results
                results_dict['image'] = result.orig_img
            else:
                # If image is an array, use it directly
                results_dict['image'] = image

        return results_dict

    except Exception as e:
        raise ModelInferenceError(f"Error during framing detection: {str(e)}")

def train(
    self,
    data_yaml: Union[str, Path],
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    project: str = 'framing_detection',
    name: str = 'train',
    device: Optional[str] = None,
    lr0: float = 0.01,
    lrf: float = 0.01,
    save: bool = True,
    resume: bool = False,
    pretrained: bool = True,
    **kwargs
) -> Any:
    """
    Train the detector on a dataset.

    Args:
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        patience: Epochs to wait for no improvement before early stopping
        project: Project name for saving results
        name: Run name for this training session
        device: Device to use (None for auto-detection)
        lr0: Initial learning rate
        lrf: Final learning rate (fraction of lr0)
        save: Whether to save the model
        resume: Resume training from the last checkpoint
        pretrained: Use pretrained weights
        **kwargs: Additional arguments to pass to the trainer

    Returns:
        Training results
    """
    device = device or self.device

    logger.info(f"Training YOLOv8{self.model_size} on {device}")
    logger.info(f"Data config: {data_yaml}, Epochs: {epochs}, Batch size: {batch_size}")

    # Set up training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,
        'project': project,
        'name': name,
        'device': device,
        'lr0': lr0,
        'lrf': lrf,
        'save': save,
        'pretrained': pretrained,
        'resume': resume
    }

    # Add any additional kwargs
    train_args.update(kwargs)

    # Start training
    try:
        results = self.model.train(**train_args)

        logger.info(f"Training completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def export(
    self,
    format: str = 'onnx',
    output_path: Optional[Union[str, Path]] = None,
    dynamic: bool = True,
    half: bool = True,
    simplify: bool = True
) -> Path:
    """
    Export the model to a deployable format.

    Args:
        format: Export format ('onnx', 'torchscript', 'openvino', etc.)
        output_path: Path to save the exported model
        dynamic: Use dynamic axes in ONNX export
        half: Export with half precision (FP16)
        simplify: Simplify the model during export

    Returns:
        Path: Path to the exported model
    """
    if output_path is None:
        # Generate default output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"framing_detector_{timestamp}.{format}"
    else:
        output_path = Path(output_path)

    logger.info(f"Exporting model to {format} format: {output_path}")

    try:
        # Export the model
        exported_path = self.model.export(
            format=format,
            imgsz=self.img_size,
            dynamic=dynamic,
            half=half,
            simplify=simplify
        )

        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If the export path is different from the desired output path, move it
        if str(exported_path) != str(output_path):
            shutil.copy(exported_path, output_path)
            os.remove(exported_path)
            logger.info(f"Moved exported model to {output_path}")

        logger.info(f"Model exported successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise

def save_checkpoint(
    self,
    path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> Path:
    """
    Save a model checkpoint.

    Args:
        path: Path to save the checkpoint
        overwrite: Whether to overwrite if file exists

    Returns:
        Path: Path to the saved checkpoint
    """
    if path is None:
        # Generate default path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = CHECKPOINTS_DIR / f"framing_detector_{timestamp}.pt"
    else:
        path = Path(path)

    # Check if file exists and overwrite is False
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint file already exists: {path}")

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model checkpoint to {path}")

    try:
        self.model.save(str(path))
        logger.info(f"Model checkpoint saved successfully")
        return path
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

@classmethod
def from_checkpoint(
    cls,
    checkpoint_path: Union[str, Path],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> 'FramingDetector':
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        FramingDetector: Loaded model
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise ModelNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Create detector with custom weights
    detector = cls(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        pretrained=False,
        weights_path=checkpoint_path
    )

    return detector

print(f"mAP: {metrics['mAP']:.4f}")

"""""

This implementation provides a complete and flexible framing member detection system based on YOLOv8. The model can identify various framing elements in residential construction images, which is essential for the electrician time estimation application. The code includes comprehensive training, evaluation, and inference capabilities, along with utilities for saving, loading, and exporting models.





Prompt 4


# Measurement Estimation Module for Framing Member Detection

This module provides comprehensive functionality for estimating real-world measurements from detected framing members in residential construction images.

## Directory Structure


models/
└── measurements/
├── __init__.py
├── measurement_estimator.py
├── reference_scale.py
├── spacing_calculator.py
├── dimension_estimator.py
├── path_calculator.py
└── visualization.py

"""
## Implementation Files

### models/measurements/\_\_init\_\_.py

""""
Measurement estimation module for the electrician time estimation application.
This module provides tools for calculating distances, dimensions, and wiring paths
from detected framing members in residential construction images.
"""

from models.measurements.measurement_estimator import MeasurementEstimator
from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from models.measurements.visualization import (
visualize_measurements,
visualize_wiring_path,
visualize_scale_calibration
)

__all__ = [
'MeasurementEstimator',
'ReferenceScale',
'ScaleCalibration',
'SpacingCalculator',
'DimensionEstimator',
'PathCalculator',
'visualize_measurements',
'visualize_wiring_path',
'visualize_scale_calibration'
]


### models/measurements/measurement_estimator.py

"""
Main measurement estimation class for analyzing framing members.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json

from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("measurement_estimator")

class MeasurementEstimator:
    """
    Class for estimating measurements from detected framing members.
    """

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None,
        confidence_threshold: float = 0.7,
        detection_threshold: float = 0.25
    ):
        """
        Initialize the measurement estimator.

        Args:
            pixels_per_inch: Calibration value (pixels per inch)
            calibration_data: Pre-computed calibration data
            confidence_threshold: Threshold for including detections in measurements
            detection_threshold: Threshold for detection confidence
        """
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold

        # Initialize the reference scale
        self.reference_scale = ReferenceScale(
            pixels_per_inch=pixels_per_inch,
            calibration_data=calibration_data
        )

        # Initialize measurement components
        self.spacing_calculator = SpacingCalculator(self.reference_scale)
        self.dimension_estimator = DimensionEstimator(self.reference_scale)
        self.path_calculator = PathCalculator(self.reference_scale)

        # Store measurement history
        self.last_measurement_result = None

        logger.info("Measurement estimator initialized")

    def calibrate_from_reference(
        self,
        image: np.ndarray,
        reference_points: List[Tuple[int, int]],
        reference_distance: float,
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using reference points.

        Args:
            image: Input image array
            reference_points: List of two (x, y) points defining a known distance
            reference_distance: Known distance between points
            units: Units of the reference distance ("inches", "feet", "mm", "cm", "m")

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_points(
                reference_points, reference_distance, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def calibrate_from_known_object(
        self,
        image: np.ndarray,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using a known object.

        Args:
            image: Input image array
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            units: Units of the reference dimensions

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_object(
                object_bbox, object_dimensions, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def analyze_framing_measurements(
        self,
        detection_result: Dict,
        calibration_check: bool = True
    ) -> Dict:
        """
        Analyze framing member detections to extract measurements.

        Args:
            detection_result: Detection results from framing detector
            calibration_check: Whether to verify calibration first

        Returns:
            Dict: Measurement analysis results
        """
        if calibration_check and not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Filter detections based on confidence
            detections = [det for det in detection_result['detections']
                          if det['confidence'] >= self.detection_threshold]

            if not detections:
                logger.warning("No valid detections found for measurement analysis")
                return {
                    "status": "warning",
                    "message": "No valid detections found",
                    "measurements": {}
                }

            # Extract image if available
            image = detection_result.get('image')

            # Calculate spacing measurements
            spacing_results = self.spacing_calculator.calculate_spacings(detections, image)

            # Estimate framing dimensions
            dimension_results = self.dimension_estimator.estimate_dimensions(detections, image)

            # Collect all measurements and calculate overall confidence
            measurements = {
                "spacing": spacing_results,
                "dimensions": dimension_results,
                "unit": self.reference_scale.get_unit(),
                "pixels_per_unit": self.reference_scale.get_pixels_per_unit()
            }

            # Calculate overall confidence score
            detection_confs = [det['confidence'] for det in detections]
            avg_detection_conf = sum(detection_confs) / len(detection_confs) if detection_confs else 0

            spacing_conf = spacing_results.get("confidence", 0) if spacing_results else 0
            dimension_conf = dimension_results.get("confidence", 0) if dimension_results else 0
            scale_conf = self.reference_scale.get_calibration_confidence()

            overall_confidence = 0.4 * avg_detection_conf + 0.3 * spacing_conf + \
                               0.2 * dimension_conf + 0.1 * scale_conf

            # Store the results for later reference
            self.last_measurement_result = {
                "status": "success",
                "message": "Measurement analysis completed",
                "measurements": measurements,
                "confidence": overall_confidence
            }

            return self.last_measurement_result

        except Exception as e:
            error_msg = f"Measurement analysis failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def estimate_wiring_path(
        self,
        detection_result: Dict,
        path_points: List[Tuple[int, int]],
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Estimate the total distance of a wiring path.

        Args:
            detection_result: Detection results from framing detector
            path_points: List of (x, y) points defining the wiring path
            drill_points: List of (x, y) points where drilling is required

        Returns:
            Dict: Wiring path analysis
        """
        if not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Extract image if available
            image = detection_result.get('image')

            # Calculate path measurements
            path_results = self.path_calculator.calculate_path(
                path_points, image, drill_points=drill_points
            )

            # Calculate drill points if not provided
            if drill_points is None and 'detections' in detection_result:
                detected_drill_points = self.path_calculator.identify_drill_points(
                    path_points, detection_result['detections'], image
                )
                path_results['detected_drill_points'] = detected_drill_points

            # Store the results
            self.last_path_result = path_results

            return path_results

        except Exception as e:
            error_msg = f"Wiring path estimation failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def save_measurements(self, output_file: Union[str, Path]) -> None:
        """
        Save the last measurement results to a file.

        Args:
            output_file: Path to save the measurement data
        """
        if self.last_measurement_result is None:
            logger.warning("No measurements to save")
            return

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w') as f:
                json.dump(self.last_measurement_result, f, indent=2)

            logger.info(f"Measurement results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save measurements: {str(e)}")

    def load_measurements(self, input_file: Union[str, Path]) -> Dict:
        """
        Load measurement results from a file.

        Args:
            input_file: Path to the measurement data file

        Returns:
            Dict: Loaded measurement data
        """
        input_file = Path(input_file)

        if not input_file.exists():
            error_msg = f"Measurement file not found: {input_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(input_file, 'r') as f:
                measurement_data = json.load(f)

            self.last_measurement_result = measurement_data

            # Update calibration if present
            if 'measurements' in measurement_data and 'pixels_per_unit' in measurement_data['measurements']:
                unit = measurement_data['measurements'].get('unit', 'inches')
                pixels_per_unit = measurement_data['measurements']['pixels_per_unit']

                self.reference_scale.set_calibration(pixels_per_unit, unit)

            logger.info(f"Measurement results loaded from {input_file}")
            return measurement_data

        except Exception as e:
            def identify_drill_points(
                self,
                path_points: List[Tuple[float, float]],
                detections: List[Dict],
                image: Optional[np.ndarray] = None
            ) -> List[Dict]:
                """
                Identify points where drilling is required along a wiring path.

                Args:
                    path_points: List of (x, y) points defining the wiring path
                    detections: List of detection dictionaries
                    image: Original image (optional, for visualization)

                Returns:
                    List[Dict]: List of drill points
                """
                if len(path_points) < 2:
                    return []

                drill_points = []

                # Process each path segment
                for i in range(len(path_points) - 1):
                    start_x, start_y = path_points[i]
                    end_x, end_y = path_points[i+1]

                    # Check intersection with each framing member
                    for det in detections:
                        # Skip non-framing categories
                        category = det['category_name']
                        if category not in ['stud', 'joist', 'rafter', 'beam', 'plate', 'header']:
                            continue

                        # Get bounding box
                        bbox = det['bbox']
                        x, y, w, h = bbox

                        # Check if segment intersects the bounding box
                        if self._segment_intersects_box(start_x, start_y, end_x, end_y, x, y, w, h):
                            # Calculate intersection point
                            intersection = self._get_segment_box_intersection(
                                start_x, start_y, end_x, end_y, x, y, w, h
                            )

                            if intersection:
                                # Determine drill difficulty based on member type and size
                                difficulty = self._calculate_drill_difficulty(det)

                                drill_points.append({
                                    "position": intersection,
                                    "requires_drilling": True,
                                    "category": category,
                                    "difficulty": difficulty
                                })

                return drill_points

            def _segment_intersects_box(
                self,
                start_x: float,
                start_y: float,
                end_x: float,
                end_y: float,
                box_x: float,
                box_y: float,
                box_width: float,
                box_height: float
            ) -> bool:
                """
                Check if a line segment intersects with a bounding box.

                Args:
                    start_x, start_y: Start point of segment
                    end_x, end_y: End point of segment
                    box_x, box_y, box_width, box_height: Bounding box

                Returns:
                    bool: True if the segment intersects the box
                """
                # Define box corners
                left = box_x
                right = box_x + box_width
                top = box_y
                bottom = box_y + box_height

                # Check if either endpoint is inside the box
                if (left <= start_x <= right and top <= start_y <= bottom) or \
                   (left <= end_x <= right and top <= end_y <= bottom):
                    return True

                # Check if line segment intersects any of the box edges
                edges = [
                    (left, top, right, top),          # Top edge
                    (right, top, right, bottom),      # Right edge
                    (left, bottom, right, bottom),    # Bottom edge
                    (left, top, left, bottom)         # Left edge
                ]

                for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
                    if self._line_segments_intersect(
                        start_x, start_y, end_x, end_y,
                        edge_x1, edge_y1, edge_x2, edge_y2
                    ):
                        return True

                return False

            def _line_segments_intersect(
                self,
                a_x1: float, a_y1: float, a_x2: float, a_y2: float,
                b_x1: float, b_y1: float, b_x2: float, b_y2: float
            ) -> bool:
                """
                Check if two line segments intersect.

                Args:
                    a_x1, a_y1, a_x2, a_y2: First line segment
                    b_x1, b_y1, b_x2, b_y2: Second line segment

                Returns:
                    bool: True if the segments intersect
                """
                # Calculate the direction vectors
                r = (a_x2 - a_x1, a_y2 - a_y1)
                s = (b_x2 - b_x1, b_y2 - b_y1)

                # Calculate the cross product (r × s)
                rxs = r[0] * s[1] - r[1] * s[0]

                # If r × s = 0, the lines are collinear or parallel
                if abs(rxs) < 1e-8:
                    return False

                # Calculate t and u parameters
                q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
                t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
                u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

                # Check if intersection point is within both segments
                return 0 <= t <= 1 and 0 <= u <= 1

            def _get_segment_box_intersection(
                self,
                start_x: float,
                start_y: float,
                end_x: float,
                end_y: float,
                box_x: float,
                box_y: float,
                box_width: float,
                box_height: float
            ) -> Optional[Tuple[float, float]]:
                """
                Get the intersection point of a line segment with a box.

                Args:
                    start_x, start_y: Start point of segment
                    end_x, end_y: End point of segment
                    box_x, box_y, box_width, box_height: Bounding box

                Returns:
                    Optional[Tuple[float, float]]: Intersection point or None
                """
                # Define box corners
                left = box_x
                right = box_x + box_width
                top = box_y
                bottom = box_y + box_height

                # If start point is inside the box, use it
                if left <= start_x <= right and top <= start_y <= bottom:
                    return (start_x, start_y)

                # If end point is inside the box, use it
                if left <= end_x <= right and top <= end_y <= bottom:
                    return (end_x, end_y)

                # Check intersections with box edges
                edges = [
                    (left, top, right, top),          # Top edge
                    (right, top, right, bottom),      # Right edge
                    (left, bottom, right, bottom),    # Bottom edge
                    (left, top, left, bottom)         # Left edge
                ]

                for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
                    if self._line_segments_intersect(
                        start_x, start_y, end_x, end_y,
                        edge_x1, edge_y1, edge_x2, edge_y2
                    ):
                        # Calculate intersection point
                        intersection = self._calculate_intersection_point(
                            start_x, start_y, end_x, end_y,
                            edge_x1, edge_y1, edge_x2, edge_y2
                        )

                        if intersection:
                            return intersection

                return None

            def _calculate_intersection_point(
                self,
                a_x1: float, a_y1: float, a_x2: float, a_y2: float,
                b_x1: float, b_y1: float, b_x2: float, b_y2: float
            ) -> Optional[Tuple[float, float]]:
                """
                Calculate the intersection point of two line segments.

                Args:
                    a_x1, a_y1, a_x2, a_y2: First line segment
                    b_x1, b_y1, b_x2, b_y2: Second line segment

                Returns:
                    Optional[Tuple[float, float]]: Intersection point or None
                """
                # Calculate the direction vectors
                r = (a_x2 - a_x1, a_y2 - a_y1)
                s = (b_x2 - b_x1, b_y2 - b_y1)

                # Calculate the cross product (r × s)
                rxs = r[0] * s[1] - r[1] * s[0]

                # If r × s = 0, the lines are collinear or parallel
                if abs(rxs) < 1e-8:
                    return None

                # Calculate t parameter
                q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
                t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
                u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

                # Check if intersection point is within both segments
                if 0 <= t <= 1 and 0 <= u <= 1:
                    # Calculate intersection point
                    ix = a_x1 + t * r[0]
                    iy = a_y1 + t * r[1]
                    return (ix, iy)

                return None

            def _calculate_drill_difficulty(self, detection: Dict) -> str:
                """
                Estimate the difficulty of drilling through a detected member.

                Args:
                    detection: Detection dictionary

                Returns:
                    str: Difficulty level ("easy", "moderate", "difficult")
                """
                category = detection['category_name']

                # Estimate based on category
                if category in ['stud', 'plate']:
                    base_difficulty = "easy"
                elif category in ['joist', 'rafter']:
                    base_difficulty = "moderate"
                elif category in ['beam', 'header']:
                    base_difficulty = "difficult"
                else:
                    base_difficulty = "moderate"

                # Adjust based on member size if available
                if 'dimensions' in detection:
                    thickness = detection['dimensions'].get('thickness', 0)

                    # Convert to inches for standard comparison
                    if self.reference_scale.get_unit() != "inches":
                        thickness_inches = self.reference_scale.convert_units(
                            thickness, self.reference_scale.get_unit(), "inches"
                        )
                    else:
                        thickness_inches = thickness

                    # Adjust difficulty based on thickness
                    if thickness_inches > 3.0:  # Thick member
                        if base_difficulty == "easy":
                            base_difficulty = "moderate"
                        elif base_difficulty == "moderate":
                            base_difficulty = "difficult"
                    elif thickness_inches < 1.0:  # Thin member
                        if base_difficulty == "difficult":
                            base_difficulty = "moderate"
                        elif base_difficulty == "moderate":
                            base_difficulty = "easy"

                return base_difficulty

            def estimate_drilling_time(
                self,
                drill_points: List[Dict],
                drill_speed: str = "normal"
            ) -> Dict:
                """
                Estimate the time required for drilling through framing members.

                Args:
                    drill_points: List of drill points
                    drill_speed: Drilling speed ("slow", "normal", "fast")

                Returns:
                        Dict: Time estimates
                    """
                    if not drill_points:
                        return {
                            "total_time_minutes": 0,
                            "drill_points": 0,
                            "average_time_per_point": 0
                        }
    
                    # Base time per difficulty level (in minutes)
                    time_factors = {
                        "easy": {"slow": 5, "normal": 3, "fast": 2},
                        "moderate": {"slow": 8, "normal": 5, "fast": 3},
                        "difficult": {"slow": 12, "normal": 8, "fast": 5}
                    }
    
                    total_time = 0
    
                    for point in drill_points:
                        difficulty = point.get("difficulty", "moderate")
                        time_per_drill = time_factors.get(difficulty, time_factors["moderate"])[drill_speed]
                        total_time += time_per_drill
    
                    return {
                        "total_time_minutes": total_time,
                        "drill_points": len(drill_points),
                        "average_time_per_point": total_time / len(drill_points)
                    }
    
    def get_image_info(image_path: Union[str, Path]) -> Dict:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: Information about the image
            
        Raises:
            ImageProcessingError: If there is an error processing the image
        """
        from PIL import Image
        image_path = Path(image_path)
    
        raise ImageProcessingError(f"Error getting image info for {image_path}: {str(e)}")
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    info = {
                        'filename': image_path.name,
                        'path': str(image_path),
                        'format': img.format,
                        'mode': img.mode,
                        'width': img.width,
                        'height': img.height,
                        'size_bytes': image_path.stat().st_size,
                    }

                    # Try to get EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif() is not None:
                        info['has_exif'] = True
                    else:
                        info['has_exif'] = False

                    return info
            except Exception as e:
                # Using a generic exception since ImageProcessingError may not be defined
                raise Exception(f"Error getting image info for {image_path}: {str(e)}")

def list_images(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    List all valid images in a directory.

    Args:
        directory: Directory to search for images
        recursive: Whether to search recursively

    Returns:
        List[Path]: List of paths to valid images
    """
    import logging
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    valid_images = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return valid_images

    # Get all files
    if recursive:
        all_files = list(directory.glob('**/*'))
    else:
        all_files = list(directory.glob('*'))

    # Define valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Filter for image files
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            # Simple validation function (can be expanded)
            def validate_image(path):
                try:
                    from PIL import Image
                    with Image.open(path) as img:
                        return img.verify() is None
                except:
                    return False
            
            if validate_image(file_path):
                valid_images.append(file_path)

    return valid_images

# Example implementation for image collector 
class ImageCollector:
    """
    Class for collecting and organizing image datasets for training.
    """

    def __init__(self, target_dir=None):
        """
        Initialize the ImageCollector.

        Args:
            target_dir: Directory to store collected images
        """
        if target_dir is None:
            target_dir = Path('./data/images')
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_directory(self, source_dir,
                                 category="unclassified",
                                 recursive=True,
                                 copy=True,
                                 overwrite=False):
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
    def collect_from_directory(self, source_dir,
                                 category="unclassified",
                                 recursive=True,
                                 copy=True,
                                 overwrite=False):
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
            overwrite: Whether to overwrite existing files

        Returns:
            Dict: Statistics about the collection process
        """
        import logging
        logger = logging.getLogger(__name__)
        

            # Try to get EXIF data if available
            if hasattr(img, '_getexif') and img._getexif() is not None:
                info['has_exif'] = True
            else:
                info['has_exif'] = False

            return info
    except Exception as e:
        raise ImageProcessingError(f"Error getting image info for {image_path}: {str(e)}")

def list_images(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    List all valid images in a directory.

    Args:
        directory: Directory to search for images
        recursive: Whether to search recursively

    Returns:
        List[Path]: List of paths to valid images
    """
    from pathlib import Path
    import logging
    
    # Define valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    valid_images = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return valid_images

    # Get all files
    if recursive:
        all_files = list(directory.glob('**/*'))
    else:
        all_files = list(directory.glob('*'))

    # Filter for image files
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            # Use validate_image function which should be defined elsewhere
            if validate_image(file_path):
                valid_images.append(file_path)

    return valid_images

# The remaining code sections were causing errors and are commented out
"""
# data/collectors/image_collector.py


#Module for collecting and organizing images for the electrician time estimation application.


import os
import shutil
from pathlib import Path
from typing import Union, List, Optional, Dict
import uuid
import hashlib

from config.settings import IMAGES_DIR
from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import validate_image, get_image_info, list_images

logger = get_logger("image_collector")

class ImageCollector:
    """
    Class for collecting and organizing image datasets for training.
    """

    def __init__(self, target_dir: Union[str, Path] = IMAGES_DIR):
        """
        Initialize the ImageCollector.

        Args:
            target_dir: Directory to store collected images
        """
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_directory(self, source_dir: Union[str, Path],
                                 category: str = "unclassified",
                                 recursive: bool = True,
                                 copy: bool = True,
                                 overwrite: bool = False) -> Dict:
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
            overwrite: Whether to overwrite existing files

        Returns:
            Dict: Statistics about the collection process
        """
        source_dir = Path(source_dir)

        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # Create category subdirectory if needed
        category_dir = self.target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Get list of valid images
        image_paths = list_images(source_dir, recursive=recursive)

        stats = {
            "total_found": len(image_paths),
            "successfully_collected": 0,
            "failed": 0,
            "skipped": 0
        }

        for img_path in image_paths:
            try:
                # Generate a unique filename based on content hash + original name
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:10]

                new_filename = f"{file_hash}_{img_path.name}"
                dest_path = category_dir / new_filename

                # Check if file already exists
                if dest_path.exists() and not overwrite:
                    logger.info(f"Skipping existing file: {dest_path}")
                    stats["skipped"] += 1
                    continue

                # Copy or move the file
                if copy:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(img_path, dest_path)

                stats["successfully_collected"] += 1
                logger.debug(f"Collected image: {dest_path}")

            except Exception as e:
                logger.error(f"Failed to collect image {img_path}: {str(e)}")
                stats["failed"] += 1

        logger.info(f"Collection completed. Stats: {stats}")
        return stats

    def collect_single_image(self, image_path: Union[str, Path],
                                  category: str = "unclassified",
                                  new_name: Optional[str] = None,
                                  copy: bool = True,
                                  overwrite: bool = False) -> Path:
        """
        Collect a single image and store it in an organized manner.

        Args:
            image_path: Path to the image file
            category: Category to assign to the image
            new_name: New name for the image (if None, generate a unique name)
            copy: If True, copy file; if False, move file
            overwrite: Whether to overwrite existing files

        Returns:
            Path: Path to the collected image
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        # Create category subdirectory if needed
        category_dir = self.target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if new_name is None:
            # Generate a unique filename based on content hash + original name
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:10]

            new_filename = f"{file_hash}_{image_path.name}"
        else:
            # Ensure the new name has the correct extension
            new_filename = f"{new_name}{image_path.suffix}"

        dest_path = category_dir / new_filename

        # Check if file already exists
        if dest_path.exists() and not overwrite:
            logger.info(f"File already exists: {dest_path}")
            return dest_path

        # Copy or move the file
        try:
            if copy:
                shutil.copy2(image_path, dest_path)
            else:
                shutil.move(image_path, dest_path)

            logger.info(f"Collected image: {dest_path}")
            return dest_path

        except Exception as e:
            raise ImageProcessingError(f"Failed to collect image {image_path}: {str(e)}")

    def create_dataset_index(self, output_file: Optional[Union[str, Path]] = None) -> Dict:
        """
        Create an index of all collected images.

        Args:
            output_file: Path to save the index (JSON format)

        Returns:
            Dict: Dataset index
        """
        index = {
            "total_images": 0,
            "categories": {}
        }

        # Scan through the target directory
        for category_dir in [d for d in self.target_dir.iterdir() if d.is_dir()]:
            category_name = category_dir.name
            image_paths = list_images(category_dir, recursive=False)

            category_data = {
                "count": len(image_paths),
                "images": []
            }

            for img_path in image_paths:
                try:
                    img_info = get_image_info(img_path)
                    category_data["images"].append({
                        "filename": img_path.name,
                        "path": str(img_path.relative_to(self.target_dir)),
                        "width": img_info.get('width'),
                        "height": img_info.get('height'),
                        "size_bytes": img_info.get('size_bytes')
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for {img_path}: {str(e)}")

            index["categories"][category_name] = category_data
            index["total_images"] += category_data["count"]

        # Save index to file if specified
        if output_file is not None:
            import json
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(index, f, indent=2)

            logger.info(f"Dataset index saved to {output_file}")

        return index
    
# Example implementation for image processor
# data/processors/image_processor.py
"""
Image processing utilities for the electrician time estimation application.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any
import os

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("image_processor")

class ImageProcessor:
    """
    Class for processing images of framing members.
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImageProcessor.

        Args:
            target_size: Target size (width, height) for processed images
        """
        self.target_size = target_size

    def preprocess_image(self,
                           image: Union[str, Path, np.ndarray],
                           normalize: bool = True,
                           enhance_contrast: bool = False,
                           denoise: bool = False) -> np.ndarray:
        """
        Preprocess an image for analysis or model input.

        Args:
            image: Image file path or numpy array
            normalize: Whether to normalize pixel values to [0,1]
            enhance_contrast: Whether to enhance image contrast
            denoise: Whether to apply denoising

        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Apply denoising if requested
        if denoise:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Enhance contrast if requested
        if enhance_contrast:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Resize if target size is specified
        if self.target_size is not None:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0

        return img

    def batch_process(self,
                        image_paths: List[Union[str, Path]],
                        output_dir: Union[str, Path],
                        preprocessing_params: Dict[str, Any] = None) -> List[Path]:
        """
        Process a batch of images and save the results.

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save processed images
            preprocessing_params: Parameters for preprocessing

        Returns:
            List[Path]: Paths to processed images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if preprocessing_params is None:
            preprocessing_params = {
                'normalize': False, # Don't normalize for saved images
                'enhance_contrast': True,
                'denoise': True
            }

        processed_paths = []

        for img_path in image_paths:
            try:
                img_path = Path(img_path)

                if not validate_image(img_path):
                    logger.warning(f"Skipping invalid image: {img_path}")
                    continue

                # Process the image
                processed_img = self.preprocess_image(img_path, **preprocessing_params)

                # For saving, convert back to uint8 if normalized
                if preprocessing_params.get('normalize', False):
                    processed_img = (processed_img * 255).astype(np.uint8)

                # Save the processed image
                output_path = output_dir / img_path.name
                save_image(processed_img, output_path)
                processed_paths.append(output_path)

                logger.debug(f"Processed image: {output_path}")

            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {str(e)}")

        logger.info(f"Batch processing completed. Processed {len(processed_paths)} images.")
        return processed_paths

    def extract_features(self,
                         image: Union[str, Path, np.ndarray],
                         feature_type: str = 'edges') -> np.ndarray:
        """
        Extract features from an image.

        Args:
            image: Image file path or numpy array
            feature_type: Type of features to extract ('edges', 'corners', 'lines')

        Returns:
            np.ndarray: Feature image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image, color_mode='grayscale')
        elif len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()

        if feature_type == 'edges':
            # Edge detection (good for framing members)
            features = cv2.Canny(img, 50, 150)
        elif feature_type == 'corners':
            # Corner detection
            corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
            features = np.zeros_like(img)
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(features, (int(x), int(y)), 3, 255, -1)
        elif feature_type == 'lines':
            # Line detection (good for framing members)
            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            features = np.zeros_like(img)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(features, (x1, y1), (x2, y2), 255, 2)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return features

    def analyze_framing(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """
        Basic analysis of framing members in an image.

        Args:
            image: Image file path or numpy array

        Returns:
            Dict: Analysis results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
            original_img = img.copy()
        else:
            img = image.copy()
            original_img = img.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        results = {
            "has_framing_lines": lines is not None and len(lines) > 0,
            "line_count": 0 if lines is None else len(lines),
            "orientation_stats": {"horizontal": 0, "vertical": 0, "diagonal": 0},
            "visualization": None
        }

        if lines is not None:
            # Create visualization image
            viz_img = original_img.copy()

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle to determine orientation
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Classify line orientation
                if angle < 20 or angle > 160:
                    results["orientation_stats"]["horizontal"] += 1
                    color = (0, 255, 0) # Green for horizontal
                elif angle > 70 and angle < 110:
                    results["orientation_stats"]["vertical"] += 1
                    color = (255, 0, 0) # Red for vertical
                else:
                    results["orientation_stats"]["diagonal"] += 1
                    color = (0, 0, 255) # Blue for diagonal

                # Draw line on visualization
                cv2.line(viz_img, (x1, y1), (x2, y2), color, 2)

            # Store visualization
            results["visualization"] = viz_img

        return results

# data/processors/augmentation.py
"""
Image augmentation for the electrician time estimation application.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import os
import json

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("augmentation")

class ImageAugmenter:
    """
    Class for augmenting images of framing members for training.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the ImageAugmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

        # Define transformations suitable for residential framing images
        self.basic_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
        ])

        # More aggressive transformations
        self.advanced_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.GaussianBlur(blur_limit=(3, 9), p=0.4),
            A.GaussNoise(var_limit=(10, 80), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.8),
            A.RandomShadow(p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
        ])

        # Special transformations for lighting simulation
        self.lighting_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=0.7),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.8),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
        ])

    def augment_image(self,
                      image: Union[str, Path, np.ndarray],
                      transform_type: str = 'basic',
                      save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            image: Image file path or numpy array
            transform_type: Type of transformation ('basic', 'advanced', 'lighting')
            save_path: Path to save the augmented image

        Returns:
            np.ndarray: Augmented image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Select transformation based on type
        if transform_type == 'basic':
            transform = self.basic_transforms
        elif transform_type == 'advanced':
            transform = self.advanced_transforms
        elif transform_type == 'lighting':
            transform = self.lighting_transforms
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

        # Apply transformation
        augmented = transform(image=img)['image']

        # Save if path is provided
        if save_path is not None:
            save_image(augmented, save_path)

        return augmented

    def create_augmentation_set(self,
                                 image_paths: List[Union[str, Path]],
                                 output_dir: Union[str, Path],
                                 transform_types: List[str] = ['basic'],
                                 samples_per_image: int = 3,
                                 include_original: bool = True) -> Dict:
        """
        Create a set of augmented images from a list of original images.

        Args:
            image_paths: List of image paths to augment
            output_dir: Directory to save augmented images
            transform_types: Types of transformations to apply
            samples_per_image: Number of augmented samples to generate per image
            include_original: Whether to include original images in the output

        Returns:
            Dict: Statistics about the augmentation process
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "original_images": len(image_paths),
            "augmented_images": 0,
            "total_images": 0,
            "transform_counts": {t: 0 for t in transform_types}
        }

        # Process each image
        for img_path in image_paths:
            try:
                img_path = Path(img_path)

                if not validate_image(img_path):
                    logger.warning(f"Skipping invalid image: {img_path}")
                    continue

                # Include original if requested
                if include_original:
                    orig_save_path = output_dir / f"orig_{img_path.name}"
                    img = load_image(img_path)
                    save_image(img, orig_save_path)
                    stats["total_images"] += 1

                # Generate augmented samples
                for i in range(samples_per_image):
                    for transform_type in transform_types:
                        try:
                            # Create unique filename for augmented image
                            aug_filename = f"aug_{transform_type}_{i}_{img_path.name}"
                            aug_save_path = output_dir / aug_filename

                            # Apply augmentation and save
                            self.augment_image(img_path, transform_type=transform_type, save_path=aug_save_path)

                            stats["augmented_images"] += 1
                            stats["transform_counts"][transform_type] += 1
                            stats["total_images"] += 1

                        except Exception as e:
                            logger.error(f"Failed to augment image {img_path} with {transform_type}: {str(e)}")

            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {str(e)}")

        # Save statistics to file
        stats_file = output_dir / "augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Augmentation completed. Created {stats['augmented_images']} augmented images.")
        return stats

    def create_custom_transform(self,
                                  transform_config: Dict[str, Any]) -> A.Compose:
        """
        Create a custom augmentation transform from a configuration.

        Args:
            transform_config: Dictionary of transform parameters

        Returns:
            A.Compose: Custom augmentation pipeline
        """
        transforms = []

        # Parse configuration and create transforms
        for transform_name, params in transform_config.items():
            if transform_name == "RandomBrightnessContrast":
                transforms.append(A.RandomBrightnessContrast(**params))
            elif transform_name == "GaussianBlur":
                transforms.append(A.GaussianBlur(**params))
            elif transform_name == "GaussNoise":
                transforms.append(A.GaussNoise(**params))
            elif transform_name == "HorizontalFlip":
                transforms.append(A.HorizontalFlip(**params))
            elif transform_name == "VerticalFlip":
                transforms.append(A.VerticalFlip(**params))
            elif transform_name == "ShiftScaleRotate":
                transforms.append(A.ShiftScaleRotate(**params))
            elif transform_name == "RandomShadow":
                transforms.append(A.RandomShadow(**params))
            elif transform_name == "RandomFog":
                transforms.append(A.RandomFog(**params))
            elif transform_name == "CLAHE":
                transforms.append(A.CLAHE(**params))
            elif transform_name == "HueSaturationValue":
                transforms.append(A.HueSaturationValue(**params))
            else:
                logger.warning(f"Unsupported transform: {transform_name}")

        return A.Compose(transforms)

    def augment_with_custom_transform(self,
                                       image: Union[str, Path, np.ndarray],
                                       custom_transform: A.Compose,
                                       save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Apply a custom augmentation transform to an image.

        Args:
            image: Image file path or numpy array
            custom_transform: Custom augmentation transform
            save_path: Path to save the augmented image

        Returns:
            np.ndarray: Augmented image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Apply transformation
        augmented = custom_transform(image=img)['image']

        # Save if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(augmented, save_path)

        return augmented



# data/annotation/annotation_tools.py
"""
Annotation tools for labeling framing members in images.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os
from typing import Union, List, Dict, Optional, Tuple
from datetime import datetime
import uuid

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("annotation_tools")

class AnnotationTool:
    """
    Class for creating and managing annotations for framing member images.
    """

    # Define categories for framing members
    CATEGORIES = [
        {"id": 1, "name": "stud", "supercategory": "framing"},
        {"id": 2, "name": "joist", "supercategory": "framing"},
        {"id": 3, "name": "rafter", "supercategory": "framing"},
        {"id": 4, "name": "beam", "supercategory": "framing"},
        {"id": 5, "name": "plate", "supercategory": "framing"},
        {"id": 6, "name": "obstacle", "supercategory": "obstacle"},
        {"id": 7, "name": "electrical_box", "supercategory": "electrical"},
        {"id": 8, "name": "plumbing", "supercategory": "obstacle"},
    ]

    def __init__(self, annotation_dir: Union[str, Path]):
        """
        Initialize the AnnotationTool.

        Args:
            annotation_dir: Directory to store annotations
        """
        self.annotation_dir = Path(annotation_dir)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        # Create a category lookup for faster access
        self.category_lookup = {cat["id"]: cat for cat in self.CATEGORIES}

    def create_coco_annotation(self,
                                image_path: Union[str, Path],
                                annotations: List[Dict],
                                output_file: Optional[Union[str, Path]] = None) -> Dict:
        """
        Create a COCO format annotation for an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the COCO JSON file

        Returns:
            Dict: COCO format annotation
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        try:
            # Load image to get dimensions
            img = load_image(image_path)
            height, width = img.shape[:2]

            # Create image info
            image_id = int(uuid.uuid4().int % (2**31 - 1)) # Random positive 32-bit int
            image_info = {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Create annotation objects
            coco_annotations = []
            for i, anno in enumerate(annotations):
                # Validate category_id
                category_id = anno.get("category_id")
                if category_id not in self.category_lookup:
                    logger.warning(f"Invalid category_id: {category_id}, skipping annotation")
                    continue

                # Get bbox
                bbox = anno.get("bbox")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}, skipping annotation")
                    continue

                # Calculate segmentation from bbox
                x, y, w, h = bbox
                segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]

                # Calculate area
                area = w * h

                # Create annotation object
                annotation_obj = {
                    "id": int(uuid.uuid4().int % (2**31 - 1)),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0
                }

                coco_annotations.append(annotation_obj)

            # Create full COCO dataset structure
            coco_data = {
                "info": {
                    "description": "Electrician Time Estimator Dataset",
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "Electrician Time Estimator",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Attribution-NonCommercial",
                        "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                    }
                ],
                "categories": self.CATEGORIES,
                "images": [image_info],
                "annotations": coco_annotations
            }

            # Save to file if specified
            if output_file is not None:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    json.dump(coco_data, f, indent=2)

                logger.info(f"COCO annotation saved to {output_file}")

            return coco_data

        except Exception as e:
            raise ImageProcessingError(f"Failed to create COCO annotation for {image_path}: {str(e)}")

    def create_yolo_annotation(self,
                                image_path: Union[str, Path],
                                annotations: List[Dict],
                                output_file: Optional[Union[str, Path]] = None) -> List[str]:
        """
        Create a YOLO format annotation for an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the YOLO txt file

        Returns:
            List[str]: YOLO format annotation lines
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        try:
            # Load image to get dimensions
            img = load_image(image_path)
            img_height, img_width = img.shape[:2]

            # Create YOLO annotation lines
            yolo_lines = []

            for anno in annotations:
                # Validate category_id
                category_id = anno.get("category_id")
                if category_id not in self.category_lookup:
                    logger.warning(f"Invalid category_id: {category_id}, skipping annotation")
                    continue

                # YOLO uses 0-indexed class numbers
                yolo_class = category_id - 1

                # Get bbox in COCO format [x, y, width, height]
                bbox = anno.get("bbox")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}, skipping annotation")
                    continue

                # Convert COCO bbox to YOLO format (normalized center x, center y, width, height)
                x, y, w, h = bbox
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                # Create YOLO line
                yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)

            # Save to file if specified
            if output_file is not None:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                logger.info(f"YOLO annotation saved to {output_file}")

            return yolo_lines

        except Exception as e:
            raise ImageProcessingError(f"Failed to create YOLO annotation for {image_path}: {str(e)}")

    def visualize_annotations(self,
                                  image_path: Union[str, Path],
                                  annotations: List[Dict],
                                  output_file: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize annotations on an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the visualization

        Returns:
            np.ndarray: Image with visualized annotations
        """
        image_path = Path(image_path)

        try:
            # Load image
            img = load_image(image_path)
            vis_img = img.copy()

            # Define colors for each category (BGR format for OpenCV)
            colors = [
                (0, 255, 0), # stud (green)
                (255, 0, 0), # joist (blue)
                (0, 0, 255), # rafter (red)
                (255, 255, 0), # beam (cyan)
                (255, 0, 255), # plate (magenta)
                (0, 255, 255), # obstacle (yellow)
                (128, 0, 128), # electrical_box (purple)
                (0, 128, 128), # plumbing (brown)
            ]

            # Draw each annotation
            for anno in annotations:
                category_id = anno.get("category_id")
                bbox = anno.get("bbox")

                if category_id not in self.category_lookup or not bbox or len(bbox) != 4:
                    continue

                # Get category name and color
                category_name = self.category_lookup[category_id]["name"]
                color = colors[(category_id - 1) % len(colors)]

                # Draw bounding box
                x, y, w, h = [int(c) for c in bbox]
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)

                # Draw label background
                text_size = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_img, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)

                # Draw label text
                cv2.putText(vis_img, category_name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save visualization if specified
            if output_file is not None:
                save_image(vis_img, output_file)
                logger.info(f"Visualization saved to {output_file}")

            return vis_img

        except Exception as e:
            raise ImageProcessingError(f"Failed to visualize annotations for {image_path}: {str(e)}")

    def merge_coco_annotations(self,
                                 annotation_files: List[Union[str, Path]],
                                 output_file: Union[str, Path]) -> Dict:
        """
        Merge multiple COCO annotation files into a single dataset.

        Args:
            annotation_files: List of COCO annotation files to merge
            output_file: Path to save the merged annotation file

        Returns:
            Dict: Merged COCO dataset
        """
        merged_dataset = {
            "info": {
                "description": "Merged Electrician Time Estimator Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Electrician Time Estimator",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }
            ],
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        # Track image and annotation IDs to avoid duplicates
        image_ids = set()
        anno_ids = set()

        # Process each annotation file
        for anno_file in annotation_files:
            try:
                with open(anno_file, 'r') as f:
                    dataset = json.load(f)

                # Add images (avoid duplicates by ID)
                for img in dataset.get("images", []):
                    if img["id"] not in image_ids:
                        merged_dataset["images"].append(img)
                        image_ids.add(img["id"])

                # Add annotations (avoid duplicates by ID)
                for anno in dataset.get("annotations", []):
                    if anno["id"] not in anno_ids:
                        merged_dataset["annotations"].append(anno)
                        anno_ids.add(anno["id"])

            except Exception as e:
                logger.error(f"Failed to process annotation file {anno_file}: {str(e)}")

        # Save merged dataset
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(merged_dataset, f, indent=2)

        logger.info(f"Merged {len(annotation_files)} annotation files to {output_file}")
        logger.info(f"Merged dataset has {len(merged_dataset['images'])} images and {len(merged_dataset['annotations'])} annotations")

        return merged_dataset


# data/annotation/annotation_converter.py
"""
Utilities for converting between annotation formats.
"""

import os
import json
import numpy as np
from pathlib import Path
import cv2
from typing import Union, List, Dict, Optional, Tuple
import glob

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, validate_image

logger = get_logger("annotation_converter")

class AnnotationConverter:
    """
    Class for converting between different annotation formats.
    """

    def __init__(self, categories: Optional[List[Dict]] = None):
        """
        Initialize the AnnotationConverter.

        Args:
            categories: List of category dictionaries (optional)
                Format: [{"id": int, "name": str, "supercategory": str}, ...]
        """
        # Default categories if none provided
        self.categories = categories or [
            {"id": 1, "name": "stud", "supercategory": "framing"},
            {"id": 2, "name": "joist", "supercategory": "framing"},
            {"id": 3, "name": "rafter", "supercategory": "framing"},
            {"id": 4, "name": "beam", "supercategory": "framing"},
            {"id": 5, "name": "plate", "supercategory": "framing"},
            {"id": 6, "name": "obstacle", "supercategory": "obstacle"},
            {"id": 7, "name": "electrical_box", "supercategory": "electrical"},
            {"id": 8, "name": "plumbing", "supercategory": "obstacle"},
        ]

        # Create category lookups
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self.name_to_id = {cat["name"]: cat["id"] for cat in self.categories}

    def yolo_to_coco(self,
                     yolo_dir: Union[str, Path],
                     image_dir: Union[str, Path],
                     output_file: Union[str, Path],
                     image_ext: str = ".jpg") -> Dict:
        """
        Convert YOLO format annotations to COCO format.

        Args:
            yolo_dir: Directory containing YOLO annotation files
            image_dir: Directory containing corresponding images
            output_file: Path to save the COCO JSON file
            image_ext: Image file extension

        Returns:
            Dict: COCO format dataset
        """
        yolo_dir = Path(yolo_dir)
        image_dir = Path(image_dir)

        # Initialize COCO dataset structure
        coco_data = {
            "info": {
                "description": "Converted from YOLO format",
                "url": "",
                "version": "1.0",
                "year": 2023,
                "contributor": "Electrician Time Estimator",
                "date_created": ""
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }
            ],
            "categories": self.categories,
            "images": [],
            "annotations": []
        }

        # Find all YOLO annotation files
        yolo_files = list(yolo_dir.glob("*.txt"))

        # Initialize counters for IDs
        image_id = 0
        annotation_id = 0

        # Process each YOLO file
        for yolo_file in yolo_files:
            try:
                # Determine corresponding image path
                image_stem = yolo_file.stem
                image_path = image_dir / f"{image_stem}{image_ext}"

                if not image_path.exists():
                    # Try finding the image with different extensions
                    potential_images = list(image_dir.glob(f"{image_stem}.*"))
                    if potential_images:
                        image_path = potential_images[0]
                    else:
                        logger.warning(f"Image not found for annotation: {yolo_file}")
                        continue

                if not validate_image(image_path):
                    logger.warning(f"Invalid image file: {image_path}")
                    continue

                # Load image to get dimensions
                img = load_image(image_path)
                height, width = img.shape[:2]

                # Add image info to COCO dataset
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "date_captured": ""
                })

                # Read YOLO annotations
                with open(yolo_file, 'r') as f:
                    yolo_annotations = f.readlines()

                # Convert each YOLO annotation line to COCO format
                for line in yolo_annotations:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse YOLO line: class x_center y_center width height
                        elements = line.split()
                        if len(elements) != 5:
                            logger.warning(f"Invalid YOLO annotation format: {line}")
                            continue

                        class_id, x_center, y_center, bbox_width, bbox_height = elements

                        # Convert to float
                        class_id = int(class_id)
                        x_center = float(x_center)
                        y_center = float(y_center)
                        bbox_width = float(bbox_width)
                        bbox_height = float(bbox_height)

                        # YOLO coordinates are normalized, convert to absolute
                        abs_width = bbox_width * width
                        abs_height = bbox_height * height
                        abs_x = (x_center * width) - (abs_width / 2)
                        abs_y = (y_center * height) - (abs_height / 2)

                        # YOLO classes are 0-indexed, COCO uses the category ID
                        coco_category_id = class_id + 1

                        # Create segmentation from bbox (simple polygon)
                        segmentation = [
                            [
                                abs_x, abs_y,
                                abs_x + abs_width, abs_y,
                                abs_x + abs_width, abs_y + abs_height,
                                abs_x, abs_y + abs_height
                            ]
                        ]

                        # Add annotation to COCO dataset
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": coco_category_id,
                            "bbox": [abs_x, abs_y, abs_width, abs_height],
                            "area": abs_width * abs_height,
                            "segmentation": segmentation,
                            "iscrowd": 0
                        })

                        annotation_id += 1

                    except Exception as e:
                        logger.warning(f"Error processing annotation line: {line}. Error: {str(e)}")

                # Increment image ID for next file
                image_id += 1

            except Exception as e:
                logger.error(f"Failed to process YOLO file {yolo_file}: {str(e)}")

        # Save COCO dataset to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"Converted {len(yolo_files)} YOLO files to COCO format: {output_file}")
        logger.info(f"Dataset has {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

        return coco_data

    def coco_to_yolo(self,
                     coco_file: Union[str, Path],
                     output_dir: Union[str, Path]) -> Dict:
        """
        Convert COCO format annotations to YOLO format.

        Args:
            coco_file: Path to COCO JSON file
            output_dir: Directory to save YOLO annotation files

        Returns:
            Dict: Statistics about the conversion
        """
        coco_file = Path(coco_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load COCO dataset
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            # Create lookup dict for images
            images = {img["id"]: img for img in coco_data["images"]}

            # Track statistics
            stats = {
                "total_images": len(coco_data["images"]),
                "total_annotations": len(coco_data["annotations"]),
                "converted_images": 0,
                "converted_annotations": 0
            }

            # Group annotations by image_id
            annotations_by_image = {}
            for anno in coco_data["annotations"]:
                image_id = anno["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(anno)

            # Process each image
            for image_id, image_info in images.items():
                # Get all annotations for this image
                image_annotations = annotations_by_image.get(image_id, [])

                if not image_annotations:
                    continue

                # Get image dimensions
                img_width = image_info["width"]
                img_height = image_info["height"]

                # Create YOLO file path
                yolo_file = output_dir / f"{Path(image_info['file_name']).stem}.txt"

                # Convert annotations to YOLO format
                yolo_lines = []

                for anno in image_annotations:
                    try:
                        # Get category ID (COCO) and convert to class ID (YOLO, 0-indexed)
                        coco_category_id = anno["category_id"]
                        yolo_class_id = coco_category_id - 1

                        # Get bounding box
                        x, y, width, height = anno["bbox"]

                        # Convert to YOLO format (normalized center x, center y, width, height)
                        x_center = (x + width / 2) / img_width
                        y_center = (y + height / 2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height

                        # Create YOLO line
                        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        yolo_lines.append(yolo_line)

                        stats["converted_annotations"] += 1

                    except Exception as e:
                        logger.warning(f"Error converting annotation: {str(e)}")

                # Write YOLO file
                with open(yolo_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                stats["converted_images"] += 1

            logger.info(f"Converted COCO to YOLO format: {stats}")
            return stats

        except Exception as e:
            raise ValueError(f"Failed to convert COCO to YOLO: {str(e)}")


# data/pipeline/data_pipeline.py
"""
Data pipeline for processing images for the electrician time estimation application.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import json
import numpy as np
import cv2
from tqdm import tqdm

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image, list_images
from data.processors.image_processor import ImageProcessor
from data.processors.augmentation import ImageAugmenter

logger = get_logger("data_pipeline")

class DataPipeline:
    """
    Pipeline for processing and preparing images for model training.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataPipeline.

        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or {}

        # Initialize processors
        self.image_processor = ImageProcessor(
            target_size=self.config.get("target_size")
        )
        self.augmenter = ImageAugmenter(
            seed=self.config.get("seed", 42)
        )

    def process_dataset(self,
                           input_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           annotation_dir: Optional[Union[str, Path]] = None,
                           preprocessing: Optional[Dict[str, Any]] = None,
                           augmentation: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Process a dataset of images and annotations.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            annotation_dir: Directory containing annotations (optional)
            preprocessing: Preprocessing parameters
            augmentation: Augmentation parameters

        Returns:
            Dict: Statistics about the processing
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy annotations if provided
        if annotation_dir is not None:
            annotation_dir = Path(annotation_dir)
            output_anno_dir = output_dir / "annotations"
            output_anno_dir.mkdir(parents=True, exist_ok=True)

            # Copy all annotation files
            for anno_file in annotation_dir.glob("*"):
                if anno_file.is_file():
                    shutil.copy2(anno_file, output_anno_dir / anno_file.name)

        # Set default preprocessing parameters
        if preprocessing is None:
            preprocessing = {
                "normalize": False,
                "enhance_contrast": True,
                "denoise": True
            }

        # Set default augmentation parameters
        if augmentation is None:
            augmentation = {
                "enabled": False,
                "transform_types": ["basic"],
                "samples_per_image": 2
            }

        # Find all valid images
        image_paths = list_images(input_dir)

        # Create processed images directory
        processed_dir = output_dir / "images"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create augmented images directory if augmentation is enabled
        if augmentation.get("enabled", False):
            augmented_dir = output_dir / "augmented"
            augmented_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        stats = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "augmented_images": 0,
            "failed_images": 0
        }

        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Preprocess image
                processed_img = self.image_processor.preprocess_image(
                    img_path,
                    normalize=preprocessing.get("normalize", False),
                    enhance_contrast=preprocessing.get("enhance_contrast", True),
                    denoise=preprocessing.get("denoise", True)
                )

                # For saving, convert back to uint8 if normalized
                if preprocessing.get("normalize", False):
                    processed_img = (processed_img * 255).astype(np.uint8)

                # Save processed image
                processed_path = processed_dir / img_path.name
                save_image(processed_img, processed_path)
                stats["processed_images"] += 1

                # Perform augmentation if enabled
                if augmentation.get("enabled", False):
                    aug_samples = augmentation.get("samples_per_image", 2)
                    transform_types = augmentation.get("transform_types", ["basic"])

                    for i in range(aug_samples):
                        for transform_type in transform_types:
                            try:
                                # Create unique filename for augmented image
                                aug_filename = f"aug_{transform_type}_{i}_{img_path.name}"
                                aug_path = augmented_dir / aug_filename

                                # Apply augmentation and save
                                self.augmenter.augment_image(
                                    processed_img,
                                    transform_type=transform_type,
                                    save_path=aug_path
                                )

                                stats["augmented_images"] += 1

                            except Exception as e:
                                logger.error(f"Augmentation failed for {img_path} with {transform_type}: {str(e)}")

            except Exception as e:
                logger.error(f"Processing failed for {img_path}: {str(e)}")
                stats["failed_images"] += 1

        # Save processing statistics
        stats_file = output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset processing completed. Stats: {stats}")
        return stats

    def prepare_model_inputs(self,
                              image_paths: List[Union[str, Path]],
                              target_size: Optional[Tuple[int, int]] = None,
                              normalize: bool = True,
                              batch_size: int = 32) -> np.ndarray:
        """
        Prepare images as input for the model.

        Args:
            image_paths: List of image paths
            target_size: Target size for images (width, height)
            normalize: Whether to normalize pixel values
            batch_size: Batch size for processing

        Returns:
            np.ndarray: Batch of processed images
        """
        # Override target_size if provided
        if target_size is not None:
            self.image_processor = ImageProcessor(target_size=target_size)

        # Initialize batch array
        if target_size:
            width, height = target_size
        else:
            # Load the first image to get dimensions
            first_img = load_image(image_paths[0])
            height, width = first_img.shape[:2]

        # Determine shape based on normalization
        if normalize:
            batch = np.zeros((len(image_paths), height, width, 3), dtype=np.float32)
        else:
            batch = np.zeros((len(image_paths), height, width, 3), dtype=np.uint8)

        # Process each image
        for i, img_path in enumerate(image_paths):
            try:
                processed_img = self.image_processor.preprocess_image(
                    img_path,
                    normalize=normalize,
                    enhance_contrast=True,
                    denoise=True
                )

                batch[i] = processed_img

            except Exception as e:
                logger.error(f"Failed to prepare image {img_path}: {str(e)}")
                # Fill with zeros for failed images

        return batch


# data/pipeline/dataset_splitter.py
"""
Utilities for splitting datasets into training, validation, and test sets.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
from sklearn.model_selection import train_test_split

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import list_images

logger = get_logger("dataset_splitter")

class DatasetSplitter:
    """
    Class for splitting datasets into training, validation, and test sets.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the DatasetSplitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def split_dataset(self,
                      dataset_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                      copy_files: bool = True,
                      annotation_dir: Optional[Union[str, Path]] = None,
                      annotation_ext: str = ".txt") -> Dict:
        """
        Split a dataset into training, validation, and test sets.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)
            annotation_dir: Directory containing annotations (optional)
            annotation_ext: Extension of annotation files

        Returns:
            Dict: Statistics about the splitting process
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        # Validate split ratios
        if sum(split_ratios) != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        # Create output directories
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create 'images' subdirectory
        (d / "images").mkdir(exist_ok=True)

        # Create 'annotations' subdirectory if annotation_dir is provided
        if annotation_dir is not None:
            (d / "annotations").mkdir(exist_ok=True)

        # List all valid images
        image_paths = list_images(dataset_dir)

        if not image_paths:
            raise ValueError(f"No valid images found in {dataset_dir}")

        # Split the dataset
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_paths, temp_paths = train_test_split(
            image_paths,
            train_size=train_ratio,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths,
            train_size=val_ratio_adjusted,
            random_state=self.seed
        )

        # Track statistics
        stats = {
            "total_images": len(image_paths),
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "test_images": len(test_paths),
            "train_annotations": 0,
            "val_annotations": 0,
            "test_annotations": 0
        }

        # Helper function to copy/move files
        def transfer_files(paths, target_dir, file_type="images"):
            count = 0
            for src_path in paths:
                # Determine destination path
                if file_type == "images":
                    dst_path = target_dir / "images" / src_path.name
                else: # annotations
                    # Use the same filename as the image but with annotation extension
                    dst_path = target_dir / "annotations" / f"{src_path.stem}{annotation_ext}"

                # Skip if destination exists and is not empty
                if dst_path.exists() and dst_path.stat().st_size > 0:
                    count += 1
                    continue

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer {src_path} to {dst_path}: {str(e)}")

            return count

        # Transfer images
        transfer_files(train_paths, train_dir)
        transfer_files(val_paths, val_dir)
        transfer_files(test_paths, test_dir)

        # Transfer annotations if annotation_dir is provided
        if annotation_dir is not None:
            annotation_dir = Path(annotation_dir)

            # Helper function to find annotation file for an image
            def find_annotation(image_path):
                anno_path = annotation_dir / f"{image_path.stem}{annotation_ext}"
                if anno_path.exists():
                    return anno_path
                return None

            # Get annotation paths for each set
            train_anno_paths = [find_annotation(img) for img in train_paths]
            val_anno_paths = [find_annotation(img) for img in val_paths]
            test_anno_paths = [find_annotation(img) for img in test_paths]

            # Remove None values
            train_anno_paths = [p for p in train_anno_paths if p is not None]
            val_anno_paths = [p for p in val_anno_paths if p is not None]
            test_anno_paths = [p for p in test_anno_paths if p is not None]

            # Transfer annotations
            stats["train_annotations"] = transfer_files(train_anno_paths, train_dir, "annotations")
            stats["val_annotations"] = transfer_files(val_anno_paths, val_dir, "annotations")
            stats["test_annotations"] = transfer_files(test_anno_paths, test_dir, "annotations")

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset split completed. Stats: {stats}")
        return stats

    def create_stratified_split(self,
                                  dataset_dir: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  annotation_dir: Union[str, Path],
                                  split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                                  copy_files: bool = True) -> Dict:
        """
        Create a stratified split based on annotation categories.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing annotations in COCO or YOLO format
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        annotation_dir = Path(annotation_dir)

        # Determine annotation format (COCO or YOLO)
        coco_files = list(annotation_dir.glob("*.json"))
        is_coco = len(coco_files) > 0

        if is_coco:
            # Process COCO format
            return self._stratified_split_coco(
                dataset_dir, output_dir, annotation_dir, split_ratios, copy_files
            )
        else:
            # Process YOLO format
            return self._stratified_split_yolo(
                dataset_dir, output_dir, annotation_dir, split_ratios, copy_files
            )

    def _stratified_split_coco(self,
                                   dataset_dir: Path,
                                   output_dir: Path,
                                   annotation_dir: Path,
                                   split_ratios: Tuple[float, float, float],
                                   copy_files: bool) -> Dict:
        """
        Create a stratified split for COCO format annotations.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing COCO format annotations
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        # Find COCO annotation file
        coco_files = list(annotation_dir.glob("*.json"))
        if not coco_files:
            raise ValueError("No COCO annotation files found")

        coco_file = coco_files[0]

        # Load COCO dataset
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Create category statistics
        category_images = {}
        for annotation in coco_data.get("annotations", []):
            category_id = annotation["category_id"]
            image_id = annotation["image_id"]

            if category_id not in category_images:
                category_images[category_id] = set()

            category_images[category_id].add(image_id)

        # Get image IDs with their highest frequency category
        image_category = {}
        for category_id, image_ids in category_images.items():
            for image_id in image_ids:
                if image_id not in image_category:
                    image_category[image_id] = []
                image_category[image_id].append(category_id)

        # Create stratification labels (use the first category for simplicity)
        stratify_labels = []
        image_ids = []

        for image_id, categories in image_category.items():
            image_ids.append(image_id)
            stratify_labels.append(categories[0])

        # Split while preserving category distribution
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_ids, temp_ids, _, temp_labels = train_test_split(
            image_ids,
            stratify_labels,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.seed
        )

        # Create image ID to filename mapping
        id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

        # Create output directories
        train_dir = output_dir / "train" / "images"
        val_dir = output_dir / "val" / "images"
        test_dir = output_dir / "test" / "images"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Helper function to transfer files
        def transfer_images(image_ids, target_dir):
            count = 0
            for image_id in image_ids:
                filename = id_to_filename.get(image_id)
                if not filename:
                    logger.warning(f"No filename found for image ID {image_id}")
                    continue

                src_path = dataset_dir / filename
                dst_path = target_dir / filename

                if not src_path.exists():
                    logger.warning(f"Source image not found: {src_path}")
                    continue

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer {src_path}: {str(e)}")

            return count

        # Transfer images
        train_count = transfer_images(train_ids, train_dir)
        val_count = transfer_images(val_ids, val_dir)
        test_count = transfer_images(test_ids, test_dir)

        # Create COCO annotation files for each split
        def create_split_coco(image_ids, output_file):
            split_coco = {
                "info": coco_data.get("info", {}),
                "licenses": coco_data.get("licenses", []),
                "categories": coco_data.get("categories", []),
                "images": [],
                "annotations": []
            }

            # Add images
            for img in coco_data.get("images", []):
                if img["id"] in image_ids:
                    split_coco["images"].append(img)

            # Add annotations
            for anno in coco_data.get("annotations", []):
                if anno["image_id"] in image_ids:
                    split_coco["annotations"].append(anno)

            # Save to file
            with open(output_file, 'w') as f:
                json.dump(split_coco, f, indent=2)

            return len(split_coco["annotations"])

        # Create annotation directories
        train_anno_dir = output_dir / "train" / "annotations"
        val_anno_dir = output_dir / "val" / "annotations"
        test_anno_dir = output_dir / "test" / "annotations"

        for d in [train_anno_dir, val_anno_dir, test_anno_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create split annotations
        train_anno_count = create_split_coco(train_ids, train_anno_dir / "instances.json")
        val_anno_count = create_split_coco(val_ids, val_anno_dir / "instances.json")
        test_anno_count = create_split_coco(test_ids, test_anno_dir / "instances.json")

        # Create statistics
        stats = {
            "total_images": len(image_ids),
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "train_annotations": train_anno_count,
            "val_annotations": val_anno_count,
            "test_annotations": test_anno_count,
            "category_distribution": {
                str(cat_id): len(img_ids) for cat_id, img_ids in category_images.items()
            }
        }

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stratified split completed. Stats: {stats}")
        return stats

    def _stratified_split_yolo(self,
                                  dataset_dir: Path,
                                  output_dir: Path,
                                  annotation_dir: Path,
                                  split_ratios: Tuple[float, float, float],
                                  copy_files: bool) -> Dict:
        """
        Create a stratified split for YOLO format annotations.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing YOLO format annotations
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        # Find all YOLO annotation files
        yolo_files = list(annotation_dir.glob("*.txt"))

        if not yolo_files:
            raise ValueError("No YOLO annotation files found")

        # Parse annotations to get category distribution
        image_categories = {}

        for yolo_file in yolo_files:
            try:
                with open(yolo_file, 'r') as f:
                    lines = f.readlines()

                categories = set()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        categories.add(class_id)

                if categories:
                    image_categories[yolo_file.stem] = list(categories)

            except Exception as e:
                logger.warning(f"Failed to parse YOLO file {yolo_file}: {str(e)}")

        # Create stratification labels (use the first category for simplicity)
        image_names = list(image_categories.keys())
        stratify_labels = [cats[0] for cats in image_categories.values()]

        # Split while preserving category distribution
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_names, temp_names, _, temp_labels = train_test_split(
            image_names,
            stratify_labels,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_names, test_names = train_test_split(
            temp_names,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.seed
        )

        # Create output directories
        train_img_dir = output_dir / "train" / "images"
        val_img_dir = output_dir / "val" / "images"
        test_img_dir = output_dir / "test" / "images"

        train_anno_dir = output_dir / "train" / "annotations"
        val_anno_dir = output_dir / "val" / "annotations"
        test_anno_dir = output_dir / "test" / "annotations"

        for d in [train_img_dir, val_img_dir, test_img_dir,
                  train_anno_dir, val_anno_dir, test_anno_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Helper function to transfer files
        def transfer_files(names, img_dir, anno_dir):
            img_count = 0
            anno_count = 0

            for name in names:
                # Find and transfer image file
                image_files = list(dataset_dir.glob(f"{name}.*"))
                image_files = [f for f in image_files if validate_image(f)]

                if not image_files:
                    logger.warning(f"No valid image found for {name}")
                    continue

                img_src = image_files[0]
                img_dst = img_dir / img_src.name

                try:
                    if copy_files:
                        shutil.copy2(img_src, img_dst)
                    else:
                        shutil.move(img_src, img_dst)
                    img_count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer image {img_src}: {str(e)}")
                    continue

                # Find and transfer annotation file
                anno_src = annotation_dir / f"{name}.txt"
                if not anno_src.exists():
                    logger.warning(f"Annotation file not found: {anno_src}")
                    continue

                anno_dst = anno_dir / f"{name}.txt"

                try:
                    if copy_files:
                        shutil.copy2(anno_src, anno_dst)
                    else:
                        shutil.move(anno_src, anno_dst)
                    anno_count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer annotation {anno_src}: {str(e)}")

            return img_count, anno_count

        # Transfer files to each split
        train_img_count, train_anno_count = transfer_files(train_names, train_img_dir, train_anno_dir)
        val_img_count, val_anno_count = transfer_files(val_names, val_img_dir, val_anno_dir)
        test_img_count, test_anno_count = transfer_files(test_names, test_img_dir, test_anno_dir)

        # Calculate category distribution
        category_counts = {}
        for cats in image_categories.values():
            for cat in cats:
                if cat not in category_counts:
                    category_counts[cat] = 0
                category_counts[cat] += 1

        # Create statistics
        stats = {
            "total_images": len(image_names),
            "train_images": train_img_count,
            "val_images": val_img_count,
            "test_images": test_img_count,
            "train_annotations": train_anno_count,
            "val_annotations": val_anno_count,
            "test_annotations": test_anno_count,
            "category_distribution": {str(k): v for k, v in category_counts.items()}
        }

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stratified split completed. Stats: {stats}")
        return stats


# models/detection/__init__.py
"""
Detection models for identifying framing members in residential construction images.
"""

from models.detection.framing_detector import FramingDetector
from models.detection.train import train_model
from models.detection.evaluate import evaluate_model
from models.detection.inference import detect_framing, visualize_detections

__all__ = [
    'FramingDetector',
    'train_model',
    'evaluate_model',
    'detect_framing',
    'visualize_detections'
]


# models/detection/model_config.py
"""
Configuration for framing member detection models.
"""

from pathlib import Path

# Framing member categories
CATEGORIES = [
    {'id': 0, 'name': 'stud'},
    {'id': 1, 'name': 'joist'},
    {'id': 2, 'name': 'rafter'},
    {'id': 3, 'name': 'beam'},
    {'id': 4, 'name': 'plate'},
    {'id': 5, 'name': 'header'},
    {'id': 6, 'name': 'blocking'},
    {'id': 7, 'name': 'electrical_box'}
]

# Color mapping for visualization
CATEGORY_COLORS = {
    'stud': (0, 255, 0),      # Green
    'joist': (255, 0, 0),     # Blue
    'rafter': (0, 0, 255),    # Red
    'beam': (0, 255, 255),     # Yellow
    'plate': (255, 0, 255),    # Magenta
    'header': (255, 255, 0),   # Cyan
    'blocking': (128, 128, 0), # Olive
    'electrical_box': (128, 0, 128) # Purple
}

# Model configuration
DEFAULT_MODEL_SIZE = 'm'  # nano, small, medium, large, extra-large
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Training configuration
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 20
DEFAULT_LR = 0.01

# Default paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / 'models' / 'weights'
CHECKPOINTS_DIR = WEIGHTS_DIR / 'checkpoints'
EXPORTS_DIR = WEIGHTS_DIR / 'exports'

# Create directories if they don't exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO-specific configuration
YOLO_CONFIG = {
    'model_type': 'yolov8',
    'pretrained_weights': f'yolov8{DEFAULT_MODEL_SIZE}.pt',
    'task': 'detect',
    'num_classes': len(CATEGORIES),
    'class_names': [cat['name'] for cat in CATEGORIES]
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,    # HSV-Hue augmentation
    'hsv_s': 0.7,      # HSV-Saturation augmentation
    'hsv_v': 0.4,      # HSV-Value augmentation
    'degrees': 0.0,    # Rotation (±deg)
    'translate': 0.1,  # Translation (±fraction)
    'scale': 0.5,      # Scale (±gain)
    'shear': 0.0,      # Shear (±deg)
    'perspective': 0.0, # Perspective (±fraction), 0.0=disabled
    'flipud': 0.0,     # Flip up-down (probability)
    'fliplr': 0.5,     # Flip left-right (probability)
    'mosaic': 1.0,     # Mosaic (probability)
    'mixup': 0.0,      # Mixup (probability)
    'copy_paste': 0.0  # Copy-paste (probability)
}


# models/detection/framing_detector.py
"""
YOLOv8-based model for detecting framing members in residential construction images.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import time
from ultralytics import YOLO

from models.detection.model_config import (
    CATEGORIES, CATEGORY_COLORS, YOLO_CONFIG,
    DEFAULT_MODEL_SIZE, DEFAULT_IMG_SIZE,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    WEIGHTS_DIR, CHECKPOINTS_DIR, EXPORTS_DIR
)
from utils.logger import get_logger
from utils.exceptions import ModelInferenceError, ModelNotFoundError

logger = get_logger("framing_detector")

class FramingDetector:
    """
    A detector for framing members in residential construction images using YOLOv8.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        img_size: int = DEFAULT_IMG_SIZE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the framing detector.

        Args:
            model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
            img_size: Input image size
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            pretrained: Whether to load pretrained weights
            weights_path: Path to custom weights file
        """
        self.model_size = model_size
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Initializing FramingDetector with YOLOv8{model_size} on {self.device}")

        # Load model
        try:
            if weights_path is not None:
                # Load custom trained weights
                weights_path = Path(weights_path)
                if not weights_path.exists():
                    raise ModelNotFoundError(f"Weights file not found: {weights_path}")

                logger.info(f"Loading custom weights from {weights_path}")
                self.model = YOLO(str(weights_path))

            elif pretrained:
                # Load pretrained weights
                logger.info(f"Loading pretrained YOLOv8{model_size}")
                self.model = YOLO(f"yolov8{model_size}.pt")

                # Update number of classes if needed
                if self.model.names != YOLO_CONFIG['class_names']:
                    logger.info(f"Updating model for {len(CATEGORIES)} framing categories")
                    self.model.names = YOLO_CONFIG['class_names']
            else:
                # Initialize with random weights
                logger.info(f"Initializing YOLOv8{model_size} with random weights")
                self.model = YOLO(f"yolov8{model_size}.yaml")

            # Move model to device
            self.model.to(self.device)

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load YOLOv8 model: {str(e)}")

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        return_original: bool = False
    ) -> Dict:
        """
        Detect framing members in an image.

        Args:
            image: Image file path or numpy array
            conf_threshold: Confidence threshold for detections (overrides default)
            iou_threshold: IoU threshold for NMS (overrides default)
            return_original: Whether to include the original image in the results

        Returns:
            Dict: Detection results with keys:
                - 'detections': List of detection dictionaries
                - 'image': Original image (if return_original=True)
                - 'inference_time': Time taken for inference
        """
        # Use specified thresholds or fall back to instance defaults
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        try:
            # Track inference time
            start_time = time.time()

            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )

            inference_time = time.time() - start_time

            # Process results
            detections = []

            # Extract results from the first image (or only image)
            result = results[0]

            # Convert boxes to the desired format
            if len(result.boxes) > 0:
                # Get boxes, classes, and confidence scores
                boxes = result.boxes.xyxy.cpu().numpy() # x1, y1, x2, y2 format
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                # Format detections
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    x1, y1, x2, y2 = box

                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1

                    # Get class name
                    class_name = result.names[cls]

                    detection = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'category_id': int(cls),
                        'category_name': class_name,
                        'confidence': float(score)
                    }

                    detections.append(detection)

            # Prepare return dictionary
            results_dict = {
                'detections': detections,
                'inference_time': inference_time
            }

            # Include original image if requested
            if return_original:
                if isinstance(image, (str, Path)):
                    # If image is a path, get the processed image from results
                    results_dict['image'] = result.orig_img
                else:
                    # If image is an array, use it directly
                    results_dict['image'] = image

            return results_dict

        except Exception as e:
            raise ModelInferenceError(f"Error during framing detection: {str(e)}")

    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        patience: int = 20,
        project: str = 'framing_detection',
        name: str = 'train',
        device: Optional[str] = None,
        lr0: float = 0.01,
        lrf: float = 0.01,
        save: bool = True,
        resume: bool = False,
        pretrained: bool = True,
        **kwargs
    ) -> Any:
        """
        Train the detector on a dataset.

        Args:
            data_yaml: Path to data configuration file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Input image size
            patience: Epochs to wait for no improvement before early stopping
            project: Project name for saving results
            name: Run name for this training session
            device: Device to use (None for auto-detection)
            lr0: Initial learning rate
            lrf: Final learning rate (fraction of lr0)
            save: Whether to save the model
            resume: Resume training from the last checkpoint
            pretrained: Use pretrained weights
            **kwargs: Additional arguments to pass to the trainer

        Returns:
            Training results
        """
        device = device or self.device

        logger.info(f"Training YOLOv8{self.model_size} on {device}")
        logger.info(f"Data config: {data_yaml}, Epochs: {epochs}, Batch size: {batch_size}")

        # Set up training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'project': project,
            'name': name,
            'device': device,
            'lr0': lr0,
            'lrf': lrf,
            'save': save,
            'pretrained': pretrained,
            'resume': resume
        }

        # Add any additional kwargs
        train_args.update(kwargs)

        # Start training
        try:
            results = self.model.train(**train_args)

            logger.info(f"Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def export(
        self,
        format: str = 'onnx',
        output_path: Optional[Union[str, Path]] = None,
        dynamic: bool = True,
        half: bool = True,
        simplify: bool = True
    ) -> Path:
        """
        Export the model to a deployable format.

        Args:
            format: Export format ('onnx', 'torchscript', 'openvino', etc.)
            output_path: Path to save the exported model
            dynamic: Use dynamic axes in ONNX export
            half: Export with half precision (FP16)
            simplify: Simplify the model during export

        Returns:
            Path: Path to the exported model
        """
        if output_path is None:
            # Generate default output path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = EXPORTS_DIR / f"framing_detector_{timestamp}.{format}"
        else:
            output_path = Path(output_path)

        logger.info(f"Exporting model to {format} format: {output_path}")

        try:
            # Export the model
            exported_path = self.model.export(
                format=format,
                imgsz=self.img_size,
                dynamic=dynamic,
                half=half,
                simplify=simplify
            )

            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # If the export path is different from the desired output path, move it
            if str(exported_path) != str(output_path):
                shutil.copy(exported_path, output_path)
                os.remove(exported_path)
                logger.info(f"Moved exported model to {output_path}")

            logger.info(f"Model exported successfully to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint
            overwrite: Whether to overwrite if file exists

        Returns:
            Path: Path to the saved checkpoint
        """
        if path is None:
            # Generate default path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = CHECKPOINTS_DIR / f"framing_detector_{timestamp}.pt"
        else:
            path = Path(path)

        # Check if file exists and overwrite is False
        if path.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint file already exists: {path}")

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model checkpoint to {path}")

        try:
            self.model.save(str(path))
            logger.info(f"Model checkpoint saved successfully")
            return path
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD
    ) -> 'FramingDetector':
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            FramingDetector: Loaded model
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ModelNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Create detector with custom weights
        detector = cls(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            pretrained=False,
            weights_path=checkpoint_path
        )

        return detector


# models/detection/train.py
"""
Training module for framing member detection models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import shutil

from models.detection.framing_detector import FramingDetector
from models.detection.model_config import (
    DEFAULT_MODEL_SIZE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_IMG_SIZE, DEFAULT_PATIENCE, DEFAULT_LR,
    WEIGHTS_DIR, AUGMENTATION_CONFIG
)
from utils.logger import get_logger

logger = get_logger("train")

def create_data_yaml(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a YAML file for YOLOv8 training.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        class_names: List of class names
        output_path: Path to save the YAML file

    Returns:
        Path: Path to the created YAML file
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    # Default output path if not specified
    if output_path is None:
        output_path = train_dir.parent / "dataset.yaml"
    else:
        output_path = Path(output_path)

    # Create the data configuration
    data_dict = {
        'path': str(train_dir.parent),
        'train': str(train_dir.relative_to(train_dir.parent)),
        'val': str(val_dir.relative_to(train_dir.parent)),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    logger.info(f"Created data YAML file: {output_path}")
    return output_path

def train_model(
    data_dir: Union[str, Path],
    model_size: str = DEFAULT_MODEL_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LR,
    pretrained: bool = True,
    augmentation: Optional[Dict] = None,
    save_checkpoint: bool = True,
    export_format: Optional[str] = 'onnx',
    project_name: str = 'framing_detection',
    run_name: Optional[str] = None
) -> Tuple[FramingDetector, Dict]:
    """
    Train a framing detection model.

    Args:
        data_dir: Directory containing the dataset with train/val subdirectories
        model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        patience: Epochs to wait for no improvement before early stopping
        learning_rate: Initial learning rate
        pretrained: Use pretrained weights
        augmentation: Augmentation parameters (None for defaults)
        save_checkpoint: Whether to save the final model checkpoint
        export_format: Format to export the model (None to skip export)
        project_name: Project name for saving results
        run_name: Run name for this training session

    Returns:
        Tuple: (Trained model, Training results)
    """
    data_dir = Path(data_dir)

    # Check if data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Expected directory structure
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Training or validation directory not found in {data_dir}")

    # Find class names from annotations
    class_names = []
    if (train_dir / "annotations").exists():
        yaml_file = list((train_dir / "annotations").glob("*.yaml"))
        if yaml_file:
            with open(yaml_file[0], 'r') as f:
                class_data = yaml.safe_load(f)
                if isinstance(class_data, dict) and 'names' in class_data:
                    class_names = list(class_data['names'].values())

    # If class names not found, use defaults from model config
    if not class_names:
        from models.detection.model_config import CATEGORIES
        class_names = [cat['name'] for cat in CATEGORIES]

    # Create data YAML file
    data_yaml = create_data_yaml(train_dir, val_dir, class_names)

    # Generate run name if not provided
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    # Initialize model
    detector = FramingDetector(model_size=model_size, pretrained=pretrained)

    # Set up augmentation parameters
    aug_params = AUGMENTATION_CONFIG.copy()
    if augmentation is not None:
        aug_params.update(augmentation)

    # Train model
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size,
        patience=patience,
        project=project_name,
        name=run_name,
        lr0=learning_rate,
        augment=True,
        **aug_params
    )

    # Save checkpoint if requested
    if save_checkpoint:
        checkpoint_path = WEIGHTS_DIR / f"{project_name}_{run_name}.pt"
        detector.save_checkpoint(checkpoint_path)

    # Export model if requested
    if export_format:
        export_path = WEIGHTS_DIR / f"{project_name}_{run_name}.{export_format}"
        detector.export(format=export_format, output_path=export_path)

    return detector, results


# models/detection/evaluate.py
"""
Evaluation functions for framing member detection models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.detection.framing_detector import FramingDetector
from data.utils.image_utils import list_images
from utils.logger import get_logger

logger = get_logger("evaluate")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box in format [x, y, width, height]
        box2: Second box in format [x, y, width, height]

    Returns:
        float: IoU value
    """
    # Convert from [x, y, width, height] to [x1, y1, x2, y2]
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]

    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_precision_recall(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> Tuple[Dict, Dict]:
    """
    Calculate precision and recall for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for a true positive
        score_threshold: Minimum confidence score to consider

    Returns:
        Tuple: (Precision per class, Recall per class)
    """
    # Filter predictions by score threshold
    filtered_preds = [p for p in predictions if p['confidence'] >= score_threshold]

    # Group by category
    pred_by_class = {}
    gt_by_class = {}

    for pred in filtered_preds:
        cat_id = pred['category_id']
        if cat_id not in pred_by_class:
            pred_by_class[cat_id] = []
        pred_by_class[cat_id].append(pred)

    for gt in ground_truth:
        cat_id = gt['category_id']
        if cat_id not in gt_by_class:
            gt_by_class[cat_id] = []
        gt_by_class[cat_id].append(gt)

    # Calculate precision and recall per class
    precision = {}
    recall = {}

    for cat_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds = pred_by_class.get(cat_id, [])
        gts = gt_by_class.get(cat_id, [])

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        # Initialize tracking variables
        tp = 0  # True positives
        fp = 0  # False positives

        # Track which ground truths have been matched
        gt_matched = [False] * len(gts)

        for pred in preds:
            pred_bbox = pred['bbox']
            matched = False

            # Check if prediction matches any unmatched ground truth
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                gt_bbox = gt['bbox']
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold:
                    tp += 1
                    gt_matched[gt_idx] = True
                    matched = True
                    break

            if not matched:
                fp += 1

        # Calculate metrics
        if tp + fp > 0:
            precision[cat_id] = tp / (tp + fp)
        else:
            precision[cat_id] = 0.0

        if len(gts) > 0:
            recall[cat_id] = tp / len(gts)
        else:
            recall[cat_id] = 0.0

    return precision, recall

def calculate_average_precision(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_points: int = 11
) -> Dict:
    """
    Calculate Average Precision (AP) for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for a true positive
        num_points: Number of points to sample for the PR curve

    Returns:
        Dict: AP per class
    """
    # Group predictions by category
    pred_by_class = {}
    gt_by_class = {}

    for pred in predictions:
        cat_id = pred['category_id']
        if cat_id not in pred_by_class:
            pred_by_class[cat_id] = []
        pred_by_class[cat_id].append(pred)

    for gt in ground_truth:
        cat_id = gt['category_id']
        if cat_id not in gt_by_class:
            gt_by_class[cat_id] = []
        gt_by_class[cat_id].append(gt)

    # Calculate AP per class
    ap_per_class = {}

    for cat_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds = pred_by_class.get(cat_id, [])
        gts = gt_by_class.get(cat_id, [])

        if not gts:
            ap_per_class[cat_id] = 0.0
            continue

        if not preds:
            ap_per_class[cat_id] = 0.0
            continue

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision and recall at each detection
        tp = 0  # True positives
        fp = 0  # False positives

        # Track which ground truths have been matched
        gt_matched = [False] * len(gts)

        precisions = []
        recalls = []

        for pred_idx, pred in enumerate(preds):
            pred_bbox = pred['bbox']
            matched = False

            # Check if prediction matches any unmatched ground truth
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                gt_bbox = gt['bbox']
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold:
                    tp += 1
                    gt_matched[gt_idx] = True
                    matched = True
                    break

            if not matched:
                fp += 1

            # Calculate current precision and recall
            precision = tp / (tp + fp)
            recall = tp / len(gts)

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP by interpolating the precision-recall curve
        ap = 0.0

        # Use standard 11-point interpolation
        for t in np.linspace(0.0, 1.0, num_points):
            # Find indices where recall >= t
            indices = [i for i, r in enumerate(recalls) if r >= t]

            if indices:
                p_max = max([precisions[i] for i in indices])
                ap += p_max / num_points

        ap_per_class[cat_id] = ap

    return ap_per_class

def evaluate_model(
    model: FramingDetector,
    test_data_dir: Union[str, Path],
    annotation_dir: Optional[Union[str, Path]] = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    output_file: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Evaluate a framing detector model on test data.

    Args:
        model: The detector model to evaluate
        test_data_dir: Directory containing test images
        annotation_dir: Directory containing ground truth annotations
        iou_threshold: IoU threshold for a true positive
        conf_threshold: Confidence threshold for detections
        output_file: Path to save evaluation results

    Returns:
        Dict: Evaluation metrics
    """
    test_data_dir = Path(test_data_dir)

    # Find test images
    if (test_data_dir / "images").exists():
        test_data_dir = test_data_dir / "images"

    # Find all images
    image_paths = list_images(test_data_dir)

    if not image_paths:
        raise ValueError(f"No valid images found in {test_data_dir}")

    logger.info(f"Evaluating model on {len(image_paths)} test images")

    # Load annotations if provided
    ground_truth = []

    if annotation_dir is not None:
        annotation_dir = Path(annotation_dir)

        # Check if it's COCO format (JSON)
        coco_files = list(annotation_dir.glob("*.json"))
        if coco_files:
            # Load COCO annotations
            with open(coco_files[0], 'r') as f:
                coco_data = json.load(f)

            # Extract annotations
            for anno in coco_data.get("annotations", []):
                gt = {
                    'image_id': anno['image_id'],
                    'category_id': anno['category_id'],
                    'bbox': anno['bbox'],
                    'area': anno['area']
                }
                ground_truth.append(gt)

            # Create mapping from filename to image_id
            filename_to_id = {}
            for img in coco_data.get("images", []):
                filename_to_id[img['file_name']] = img['id']

        else:
            # Assume YOLO format (one .txt file per image)
            logger.info("No COCO annotations found, looking for YOLO format")

            # Load class names if available
            class_names = []
            yaml_files = list(annotation_dir.glob("*.yaml"))
            if yaml_files:
                with open(yaml_files[0], 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if isinstance(yaml_data, dict) and 'names' in yaml_data:
                        class_names = list(yaml_data['names'].values())

            if not class_names:
                from models.detection.model_config import CATEGORIES
                class_names = [cat['name'] for cat in CATEGORIES]

            # Process each image and its corresponding annotation
            from data.annotation.annotation_converter import AnnotationConverter
            converter = AnnotationConverter()

            for img_path in image_paths:
                img_stem = img_path.stem
                anno_path = annotation_dir / f"{img_stem}.txt"

                if not anno_path.exists():
                    logger.warning(f"No annotation found for {img_path}")
                    continue

                # Load image to get dimensions
                import cv2
                img = cv2.imread(str(img_path))
                height, width = img.shape[:2]

                # Read YOLO annotations
                with open(anno_path, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        if not line.strip():
                            continue

                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id, x_center, y_center, w, h = map(float, parts)

                        # Convert YOLO to absolute coordinates
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w = w * width
                        h = h * height

                        gt = {
                            'image_id': img_stem,
                            'category_id': int(class_id),
                            'bbox': [x, y, w, h],
                            'area': w * h
                        }
                        ground_truth.append(gt)

    # Run inference on all test images
    all_predictions = []
    total_inference_time = 0

    for img_path in tqdm(image_paths, desc="Evaluating"):
        try:
            # Run detection
            result = model.detect(
                image=str(img_path),
                conf_threshold=conf_threshold
            )

            # Get image ID (either from filename to ID mapping or use filename)
            if annotation_dir is not None and 'filename_to_id' in locals():
                img_id = filename_to_id.get(img_path.name, img_path.stem)
            else:
                img_id = img_path.stem

            # Add image ID to each detection
            for det in result['detections']:
                det['image_id'] = img_id
                all_predictions.append(det)

            total_inference_time += result['inference_time']

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

    # Calculate metrics
    metrics = {
        'num_images': len(image_paths),
        'num_predictions': len(all_predictions),
        'num_ground_truth': len(ground_truth),
        'average_inference_time': total_inference_time / len(image_paths) if image_paths else 0,
        'total_inference_time': total_inference_time
    }

    # Calculate precision and recall if ground truth is available
    if ground_truth:
        precision, recall = calculate_precision_recall(
            all_predictions, ground_truth, iou_threshold, conf_threshold
        )

        ap = calculate_average_precision(
            all_predictions, ground_truth, iou_threshold
        )

        # Calculate mAP
        if ap:
            mAP = sum(ap.values()) / len(ap)
        else:
            mAP = 0.0

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['AP'] = ap
        metrics['mAP'] = mAP

    # Save results if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation results saved to {output_file}")

    return metrics

def plot_precision_recall_curve(
    precision: Dict[int, float],
    recall: Dict[int, float],
    class_names: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot precision-recall curves.

    Args:
        precision: Precision values per class
        recall: Recall values per class
        class_names: List of class names
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Get class names if not provided
    if class_names is None:
        from models.detection.model_config import CATEGORIES
        class_names = [cat['name'] for cat in CATEGORIES]

    # Plot each class
    for class_id in precision.keys():
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"

        plt.plot(recall[class_id], precision[class_id], label=f"{class_name} (AP={precision[class_id]:.2f})")

    # Add mean average precision
    map_value = sum(precision.values()) / len(precision) if precision else 0
    plt.title(f"Precision-Recall Curve (mAP = {map_value:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best")
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
        logger.info(f"Precision-recall curve saved to {output_file}")
    else:
        plt.show()


# models/detection/inference.py
"""
Inference utilities for framing member detection models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import time

from models.detection.framing_detector import FramingDetector
from models.detection.model_config import CATEGORY_COLORS
from utils.logger import get_logger

logger = get_logger("inference")

def detect_framing(
    detector: FramingDetector,
    image_path: Union[str, Path, np.ndarray],
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None
) -> Dict:
    """
    Detect framing members in an image.

    Args:
        detector: The framing detector model
        image_path: Path to image or image array
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        Dict: Detection results
    """
    logger.debug(f"Detecting framing in image: {image_path if isinstance(image_path, (str, Path)) else 'array'}")

    return detector.detect(
        image=image_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        return_original=True
    )

def visualize_detections(
    detection_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize detection results on an image.

    Args:
        detection_result: Detection results from detect_framing
        output_path: Path to save the visualization
        show_confidence: Whether to show confidence scores
        line_thickness: Thickness of bounding box lines
        font_scale: Size of font for labels

    Returns:
        np.ndarray: Image with visualized detections
    """
    # Make sure image is included in results
    if 'image' not in detection_result:
        raise ValueError("Detection result must include the 'image' key")

    # Get image and detections
    image = detection_result['image'].copy()
    detections = detection_result['detections']

    # Draw each detection
    for det in detections:
        # Get bounding box
        x, y, w, h = [int(v) for v in det['bbox']]

        # Get category and color
        category_name = det['category_name']
        color = CATEGORY_COLORS.get(category_name, (0, 255, 0))  # Default to green

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, line_thickness)

        # Prepare label text
        if show_confidence:
            label = f"{category_name}: {det['confidence']:.2f}"
        else:
            label = category_name

        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        cv2.rectangle(image, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)

        # Draw label
        cv2.putText(
            image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), line_thickness // 2
        )

    # Save image if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Visualization saved to {output_path}")

    return image

def batch_inference(
    detector: FramingDetector,
    image_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    conf_threshold: Optional[float] = None,
    visualize: bool = True
) -> List[Dict]:
    """
    Run inference on a batch of images.

    Args:
        detector: The framing detector model
        image_paths: List of paths to images
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold for detections
        visualize: Whether to create and save visualizations

    Returns:
        List[Dict]: Detection results for each image
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for img_path in image_paths:
        img_path = Path(img_path)

        try:
            # Run detection
            result = detect_framing(
                detector=detector,
                image_path=img_path,
                conf_threshold=conf_threshold
            )

            # Save result
            results.append(result)

            # Create visualization if requested
            if visualize and output_dir is not None:
                vis_path = output_dir / f"vis_{img_path.name}"
                visualize_detections(result, output_path=vis_path)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

    return results

def analyze_framing_members(
    detection_result: Dict,
    spacing_tolerance: float = 0.1
) -> Dict:
    """
    Analyze framing members to extract structural information.

    Args:
        detection_result: Detection results from detect_framing
        spacing_tolerance: Tolerance for spacing consistency (as fraction)

    Returns:
        Dict: Analysis results
    """
    detections = detection_result['detections']

    # Separate detections by category
    by_category = {}
    for det in detections:
        cat_name = det['category_name']
        if cat_name not in by_category:
            by_category[cat_name] = []
        by_category[cat_name].append(det)

    # Analyze each category
    analysis = {
        'member_counts': {cat: len(dets) for cat, dets in by_category.items()},
        'spacing_analysis': {},
        'orientation': {}
    }

    # Function to calculate spacing between parallel members
    def analyze_spacing(members, horizontal=True):
        # Sort members by position
        if horizontal:
            # For horizontal members (like joists), sort by y-coordinate
            sorted_members = sorted(members, key=lambda x: x['bbox'][1])
        else:
            # For vertical members (like studs), sort by x-coordinate
            sorted_members = sorted(members, key=lambda x: x['bbox'][0])

        if len(sorted_members) < 2:
            return None

        # Calculate spacing between adjacent members
        spacings = []
        for i in range(len(sorted_members) - 1):
            if horizontal:
                # For horizontal members, measure center-to-center y distance
                y1 = sorted_members[i]['bbox'][1] + sorted_members[i]['bbox'][3] / 2
                y2 = sorted_members[i+1]['bbox'][1] + sorted_members[i+1]['bbox'][3] / 2
                spacing = abs(y2 - y1)
            else:
                # For vertical members, measure center-to-center x distance
                x1 = sorted_members[i]['bbox'][0] + sorted_members[i]['bbox'][2] / 2
                x2 = sorted_members[i+1]['bbox'][0] + sorted_members[i+1]['bbox'][2] / 2
                spacing = abs(x2 - x1)

            spacings.append(spacing)

        if not spacings:
            return None

        # Calculate statistics
        mean_spacing = sum(spacings) / len(spacings)
        min_spacing = min(spacings)
        max_spacing = max(spacings)

        # Check consistency
        variation = (max_spacing - min_spacing) / mean_spacing
        is_consistent = variation <= spacing_tolerance

        return {
            'mean_spacing': mean_spacing,
            'min_spacing': min_spacing,
            'max_spacing': max_spacing,
            'is_consistent': is_consistent,
            'variation': variation
        }

    # Analyze studs (vertical members)
    if 'stud' in by_category:
        analysis['spacing_analysis']['stud'] = analyze_spacing(by_category['stud'], horizontal=False)

    # Determine orientation
    orientations = []
    for stud in by_category['stud']:
        w, h = stud['bbox'][2], stud['bbox'][3]
        orientations.append('vertical' if h > w else 'horizontal')

    analysis['orientation']['stud'] = {
        'vertical': orientations.count('vertical'),
        'horizontal': orientations.count('horizontal')
    }

    # Analyze joists (horizontal members)
    if 'joist' in by_category:
        analysis['spacing_analysis']['joist'] = analyze_spacing(by_category['joist'], horizontal=True)

    # Determine orientation
    orientations = []
    for joist in by_category['joist']:
        w, h = joist['bbox'][2], joist['bbox'][3]
        orientations.append('horizontal' if w > h else 'vertical')

    analysis['orientation']['joist'] = {
        'horizontal': orientations.count('horizontal'),
        'vertical': orientations.count('vertical')
    }

    # Count other members
    for cat in by_category:
        if cat not in ('stud', 'joist'):
            analysis['member_counts'][cat] = len(by_category[cat])

    return analysis


# models/measurements/__init__.py
"""
Measurement estimation module for the electrician time estimation application.
This module provides tools for calculating distances, dimensions, and wiring paths
from detected framing members in residential construction images.
"""

from models.measurements.measurement_estimator import MeasurementEstimator
from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from models.measurements.visualization import (
    visualize_measurements,
    visualize_wiring_path,
    visualize_scale_calibration
)

__all__ = [
    'MeasurementEstimator',
    'ReferenceScale',
    'ScaleCalibration',
    'SpacingCalculator',
    'DimensionEstimator',
    'PathCalculator',
    'visualize_measurements',
    'visualize_wiring_path',
    'visualize_scale_calibration'
]


# models/measurements/measurement_estimator.py
"""
Main measurement estimation class for analyzing framing members.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json

from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("measurement_estimator")

class MeasurementEstimator:
    """
    Class for estimating measurements from detected framing members.
    """

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None,
        confidence_threshold: float = 0.7,
        detection_threshold: float = 0.25
    ):
        """
        Initialize the measurement estimator.

        Args:
            pixels_per_inch: Calibration value (pixels per inch)
            calibration_data: Pre-computed calibration data
            confidence_threshold: Threshold for including detections in measurements
            detection_threshold: Threshold for detection confidence
        """
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold

        # Initialize the reference scale
        self.reference_scale = ReferenceScale(
            pixels_per_inch=pixels_per_inch,
            calibration_data=calibration_data
        )

        # Initialize measurement components
        self.spacing_calculator = SpacingCalculator(self.reference_scale)
        self.dimension_estimator = DimensionEstimator(self.reference_scale)
        self.path_calculator = PathCalculator(self.reference_scale)

        # Store measurement history
        self.last_measurement_result = None

        logger.info("Measurement estimator initialized")

    def calibrate_from_reference(
        self,
        image: np.ndarray,
        reference_points: List[Tuple[int, int]],
        reference_distance: float,
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using reference points.

        Args:
            image: Input image array
            reference_points: List of two (x, y) points defining a known distance
            reference_distance: Known distance between points
            units: Units of the reference distance ("inches", "feet", "mm", "cm", "m")

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_points(
                reference_points, reference_distance, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def calibrate_from_known_object(
        self,
        image: np.ndarray,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using a known object.

        Args:
            image: Input image array
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            units: Units of the reference dimensions

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_object(
                object_bbox, object_dimensions, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def analyze_framing_measurements(
        self,
        detection_result: Dict,
        calibration_check: bool = True
    ) -> Dict:
        """
        Analyze framing member detections to extract measurements.

        Args:
            detection_result: Detection results from framing detector
            calibration_check: Whether to verify calibration first

        Returns:
            Dict: Measurement analysis results
        """
        if calibration_check and not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Filter detections based on confidence
            detections = [det for det in detection_result['detections']
                          if det['confidence'] >= self.detection_threshold]

            if not detections:
                logger.warning("No valid detections found for measurement analysis")
                return {
                    "status": "warning",
                    "message": "No valid detections found",
                    "measurements": {}
                }

            # Extract image if available
            image = detection_result.get('image')

            # Calculate spacing measurements
            spacing_results = self.spacing_calculator.calculate_spacings(detections, image)

            # Estimate framing dimensions
            dimension_results = self.dimension_estimator.estimate_dimensions(detections, image)

            # Collect all measurements and calculate overall confidence
            measurements = {
                "spacing": spacing_results,
                "dimensions": dimension_results,
                "unit": self.reference_scale.get_unit(),
                "pixels_per_unit": self.reference_scale.get_pixels_per_unit()
            }

            # Calculate overall confidence score
            detection_confs = [det['confidence'] for det in detections]
            avg_detection_conf = sum(detection_confs) / len(detections) if detections else 0

            spacing_conf = spacing_results.get("confidence", 0) if spacing_results else 0
            dimension_conf = dimension_results.get("confidence", 0) if dimension_results else 0
            scale_conf = self.reference_scale.get_calibration_confidence()

            overall_confidence = 0.4 * avg_detection_conf + 0.3 * spacing_conf + \
                                 0.2 * dimension_conf + 0.1 * scale_conf

            # Store the results for later reference
            self.last_measurement_result = {
                "status": "success",
                "message": "Measurement analysis completed",
                "measurements": measurements,
                "confidence": overall_confidence
            }

            return self.last_measurement_result

        except Exception as e:
            error_msg = f"Measurement analysis failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def estimate_wiring_path(
        self,
        detection_result: Dict,
        path_points: List[Tuple[int, int]],
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Estimate the total distance of a wiring path.

        Args:
            detection_result: Detection results from framing detector
            path_points: List of (x, y) points defining the wiring path
            drill_points: List of (x, y) points where drilling is required

        Returns:
            Dict: Wiring path analysis
        """
        if not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Extract image if available
            image = detection_result.get('image')

            # Calculate path measurements
            path_results = self.path_calculator.calculate_path(
                path_points, image, drill_points=drill_points
            )

            # Calculate drill points if not provided
            if drill_points is None and 'detections' in detection_result:
                detected_drill_points = self.path_calculator.identify_drill_points(
                    path_points, detection_result['detections'], image
                )
                path_results['detected_drill_points'] = detected_drill_points

            # Store the results
            self.last_path_result = path_results

            return path_results

        except Exception as e:
            error_msg = f"Wiring path estimation failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def save_measurements(self, output_file: Union[str, Path]) -> None:
        """
        Save the last measurement results to a file.

        Args:
            output_file: Path to save the measurement data
        """
        if self.last_measurement_result is None:
            logger.warning("No measurements to save")
            return

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w') as f:
                json.dump(self.last_measurement_result, f, indent=2)

            logger.info(f"Measurement results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save measurements: {str(e)}")

    def load_measurements(self, input_file: Union[str, Path]) -> Dict:
        """
        Load measurement results from a file.

        Args:
            input_file: Path to the measurement data file

        Returns:
            Dict: Loaded measurement data
        """
        input_file = Path(input_file)

        if not input_file.exists():
            error_msg = f"Measurement file not found: {input_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(input_file, 'r') as f:
                measurement_data = json.load(f)

            self.last_measurement_result = measurement_data

            # Update calibration if present
            if 'measurements' in measurement_data and 'pixels_per_unit' in measurement_data['measurements']:
                unit = measurement_data['measurements'].get('unit', 'inches')
                pixels_per_unit = measurement_data['measurements']['pixels_per_unit']

                self.reference_scale.set_calibration(pixels_per_unit, unit)

            logger.info(f"Measurement results loaded from {input_file}")
            return measurement_data

        except Exception as e:
            error_msg = f"Failed to load measurements: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)


# models/measurements/reference_scale.py
"""
Reference scale handling for measurement calibration.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass

from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("reference_scale")

@dataclass
class ScaleCalibration:
    """Scale calibration data structure."""
    pixels_per_unit: float
    unit: str
    confidence: float
    method: str
    reference_data: Dict


class ReferenceScale:
    """
    Class for handling reference scale calibration and conversion.
    """

    # Standard conversion factors to inches
    UNIT_TO_INCHES = {
        "inches": 1.0,
        "feet": 12.0,
        "mm": 0.0393701,
        "cm": 0.393701,
        "m": 39.3701
    }

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None
    ):
        """
        Initialize the reference scale.

        Args:
            pixels_per_inch: Initial calibration value
            calibration_data: Pre-computed calibration data
        """
        self.calibration = None

        # Initialize from pixels_per_inch if provided
        if pixels_per_inch is not None:
            self.calibration = ScaleCalibration(
                pixels_per_unit=pixels_per_inch,
                unit="inches",
                confidence=0.9,  # High confidence since directly provided
                method="manual",
                reference_data={"pixels_per_inch": pixels_per_inch}
            )

        # Or from calibration data if provided
        elif calibration_data is not None:
            self.calibration = ScaleCalibration(
                pixels_per_unit=calibration_data.get("pixels_per_unit", 0),
                unit=calibration_data.get("unit", "inches"),
                confidence=calibration_data.get("confidence", 0.5),
                method=calibration_data.get("method", "loaded"),
                reference_data=calibration_data.get("reference_data", {})
            )

    def is_calibrated(self) -> bool:
        """
        Check if the scale is calibrated.

        Returns:
            bool: True if calibrated
        """
        return self.calibration is not None and self.calibration.pixels_per_unit > 0

    def get_pixels_per_unit(self) -> float:
        """
        Get the current pixels per unit value.

        Returns:
            float: Pixels per unit
        """
        if not self.is_calibrated():
            return 0.0
        return self.calibration.pixels_per_unit

    def get_unit(self) -> str:
        """
        Get the current unit.

        Returns:
            str: Current unit
        """
        if not self.is_calibrated():
            return "uncalibrated"
        return self.calibration.unit

    def get_calibration_confidence(self) -> float:
        """
        Get the confidence level of the current calibration.

        Returns:
            float: Confidence level (0-1)
        """
        if not self.is_calibrated():
            return 0.0
        return self.calibration.confidence

    def get_calibration_data(self) -> Dict:
        """
        Get the full calibration data.

        Returns:
            Dict: Calibration data
        """
        if not self.is_calibrated():
            return {
                "status": "uncalibrated",
                "pixels_per_unit": 0.0,
                "unit": "uncalibrated",
                "confidence": 0.0
            }

        return {
            "status": "calibrated",
            "pixels_per_unit": self.calibration.pixels_per_unit,
            "unit": self.calibration.unit,
            "confidence": self.calibration.confidence,
            "method": self.calibration.method,
            "reference_data": self.calibration.reference_data
        }

    def set_calibration(self, pixels_per_unit: float, unit: str = "inches") -> None:
        """
        Set calibration manually.

        Args:
            pixels_per_unit: Pixels per unit value
            unit: Unit of measurement
        """
        if pixels_per_unit <= 0:
            raise ValueError("Pixels per unit must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=0.9,  # High confidence for manual setting
            method="manual",
            reference_data={"direct_setting": True}
        )

        logger.info(f"Manual calibration set: {pixels_per_unit} pixels per {unit}")

    def calibrate_from_points(
        self,
        points: List[Tuple[int, int]],
        known_distance: float,
        unit: str = "inches"
    ) -> Dict:
        """
        Calibrate from two points with a known distance.

        Args:
            points: List of two (x, y) points
            known_distance: Known distance between points
            unit: Unit of the known distance

        Returns:
            Dict: Calibration result
        """
        if len(points) != 2:
            raise ValueError("Exactly two points required for calibration")

        if known_distance <= 0:
            raise ValueError("Known distance must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        # Calculate pixel distance
        x1, y1 = points[0]
        x2, y2 = points[1]
        pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if pixel_distance <= 0:
            raise MeasurementError("Points are too close for accurate calibration")

        # Calculate pixels per unit
        pixels_per_unit = pixel_distance / known_distance

        # Set calibration
        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=0.95,  # Higher confidence for direct measurement
            method="point_distance",
            reference_data={
                "points": points,
                "known_distance": known_distance,
                "pixel_distance": pixel_distance
            }
        )

        logger.info(f"Calibrated from points: {pixels_per_unit} pixels per {unit}")

        return {
            "status": "calibrated",
            "pixels_per_unit": pixels_per_unit,
            "unit": unit,
            "confidence": 0.95,
            "method": "point_distance"
        }

    def calibrate_from_object(
        self,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        unit: str = "inches"
    ) -> Dict:
        """
        Calibrate from an object with known dimensions.

        Args:
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            unit: Unit of the known dimensions

        Returns:
            Dict: Calibration result
        """
        if len(object_bbox) != 4 or len(object_dimensions) != 2:
            raise ValueError("Invalid bounding box or dimensions format")

        if min(object_dimensions) <= 0:
            raise ValueError("Object dimensions must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        # Extract dimensions
        bbox_width, bbox_height = object_bbox[2], object_bbox[3]
        real_width, real_height = object_dimensions

        # Calculate pixels per unit for width and height
        pixels_per_unit_width = bbox_width / real_width
        pixels_per_unit_height = bbox_height / real_height

        # Average the two measurements, but weigh by the larger dimension
        # for better accuracy
        total_real = real_width + real_height
        pixels_per_unit = (
            (pixels_per_unit_width * real_width / total_real) +
            (pixels_per_unit_height * real_height / total_real)
        )

        # Calculate confidence based on aspect ratio consistency
        real_aspect = real_width / real_height if real_height != 0 else 1
        bbox_aspect = bbox_width / bbox_height if bbox_height != 0 else 1

        aspect_diff = abs(real_aspect - bbox_aspect) / max(real_aspect, bbox_aspect)
        aspect_confidence = max(0, 1 - aspect_diff)

        # Adjust confidence by object size (larger objects generally allow more accurate calibration)
        size_factor = min(1.0, max(bbox_width, bbox_height) / 300)  # Normalize to 0-1

        confidence = 0.85 * aspect_confidence + 0.15 * size_factor

        # Set calibration
        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=confidence,
            method="object_dimensions",
            reference_data={
                "object_bbox": object_bbox,
                "object_dimensions": object_dimensions,
                "pixels_per_unit_width": pixels_per_unit_width,
                "pixels_per_unit_height": pixels_per_unit_height
            }
        )

        logger.info(f"Calibrated from object: {pixels_per_unit} pixels per {unit} (confidence: {confidence:.2f})")

        return {
            "status": "calibrated",
            "pixels_per_unit": pixels_per_unit,
            "unit": unit,
            "confidence": confidence,
            "method": "object_dimensions"
        }

    def pixels_to_real_distance(self, pixels: float) -> float:
        """
        Convert pixel distance to real-world distance.

        Args:
            pixels: Distance in pixels

        Returns:
            float: Real-world distance in the calibrated unit
        """
        if not self.is_calibrated():
            raise MeasurementError("Cannot convert distance: Scale not calibrated")

        return pixels / self.calibration.pixels_per_unit

    def real_distance_to_pixels(self, distance: float) -> float:
        """
        Convert real-world distance to pixels.

        Args:
            distance: Real-world distance in the calibrated unit

        Returns:
            float: Distance in pixels
        """
        if not self.is_calibrated():
            raise MeasurementError("Cannot convert distance: Scale not calibrated")

        return distance * self.calibration.pixels_per_unit

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert a measurement between different units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            float: Converted value
        """
        if from_unit not in self.UNIT_TO_INCHES or to_unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

        # Convert to inches first, then to target unit
        inches = value * self.UNIT_TO_INCHES[from_unit]
        return inches / self.UNIT_TO_INCHES[to_unit]


# models/measurements/spacing_calculator.py
"""
Module for calculating spacings between framing members.
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("spacing_calculator")

class SpacingCalculator:
    """
    Class for calculating spacings between framing members.
    """

    def __init__(self, reference_scale: ReferenceScale):
        """
        Initialize the spacing calculator.

        Args:
            reference_scale: Reference scale for conversions
        """
        self.reference_scale = reference_scale

    def calculate_spacings(
        self,
        detections: List[Dict],
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate spacings between framing members.

        Args:
            detections: List of detection dictionaries
            image: Original image (optional, for visualization)

        Returns:
            Dict: Spacing analysis results
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        if not detections:
            return {
                "status": "error",
                "message": "No detections to analyze",
                "spacings": {}
            }

        # Group detections by category
        categories = defaultdict(list)
        for det in detections:
            cat_name = det['category_name']
            categories[cat_name].append(det)

        # Calculate spacings for each category
        results = {}
        confidence_scores = []

        # Process each category with multiple members
        for cat_name, members in categories.items():
            if len(members) < 2:
                continue

            # Get center points and analyze member orientation
            centers = []
            widths = []
            heights = []

            for member in members:
                bbox = member['bbox']
                x, y, w, h = bbox
                center_x = x + w / 2
                center_y = y + h / 2
                centers.append((center_x, center_y))
                widths.append(w)
                heights.append(h)

            # Determine if members are predominantly horizontal or vertical
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)

            if avg_height > avg_width * 1.5:
                orientation = "vertical"  # Like wall studs
            elif avg_width > avg_height * 1.5:
                orientation = "horizontal" # Like floor joists
            else:
                orientation = "mixed"

            # Sort centers based on orientation
            if orientation == "vertical":
                # Sort by x-coordinate for side-by-side spacing
                centers.sort(key=lambda p: p[0])

                # Calculate spacings between adjacent centers
                spacings_px = [centers[i+1][0] - centers[i][0] for i in range(len(centers)-1)]

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # Calculate center-to-center and clear spacings
                avg_width_px = avg_width
                avg_width_real = self.reference_scale.pixels_to_real_distance(avg_width_px)

                clear_spacings_real = [max(0, s - avg_width_real) for s in spacings_real]

            elif orientation == "horizontal":
                # Sort by y-coordinate for top-to-bottom spacing
                centers.sort(key=lambda p: p[1])

                # Calculate spacings between adjacent centers
                spacings_px = [centers[i+1][1] - centers[i][1] for i in range(len(centers)-1)]

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # Calculate center-to-center and clear spacings
                avg_height_px = avg_height
                avg_height_real = self.reference_scale.pixels_to_real_distance(avg_height_px)

                clear_spacings_real = [max(0, s - avg_height_real) for s in spacings_real]

            else:
                # For mixed orientation, calculate pairwise distances
                spacings_px = []
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = math.sqrt((centers[j][0] - centers[i][0])**2 +
                                         (centers[j][1] - centers[i][1])**2)
                        spacings_px.append(dist)

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # No clear spacing calculation for mixed orientation
                clear_spacings_real = None

            # Calculate statistics
            if spacings_real:
                mean_spacing = sum(spacings_real) / len(spacings_real)
                min_spacing = min(spacings_real)
                max_spacing = max(spacings_real)

                # Standard deviation for variability assessment
                if len(spacings_real) > 1:
                    std_dev = math.sqrt(sum((s - mean_spacing)**2 for s in spacings_real) / len(spacings_real))
                    cv = std_dev / mean_spacing if mean_spacing > 0 else float('inf') # Coefficient of variation
                else:
                    std_dev = 0
                    cv = 0

                # Determine if spacing is on standard centers
                # Convert to inches for standard comparison
                if self.reference_scale.get_unit() != "inches":
                    spacings_inches = [
                        self.reference_scale.convert_units(s, self.reference_scale.get_unit(), "inches")
                        for s in spacings_real
                    ]
                    mean_inches = sum(spacings_inches) / len(spacings_inches)
                else:
                    mean_inches = mean_spacing

                # Check if close to common framing spacings
                common_spacings = [16, 24, 12, 8, 19.2]  # Common framing centers in inches
                closest_standard = min(common_spacings, key=lambda x: abs(x - mean_inches))
                distance_to_standard = abs(closest_standard - mean_inches)

                is_standard = distance_to_standard < 1.5 # Within 1.5 inches

                # Calculate confidence based on variability and standard matching
                consistency_conf = max(0, 1 - min(cv * 2, 1)) # Lower CV = higher confidence
                standard_conf = 1.0 if is_standard else max(0, 1 - distance_to_standard / 8)

                spacing_confidence = 0.7 * consistency_conf + 0.3 * standard_conf
                confidence_scores.append(spacing_confidence)

                # Store results for this category
                results[cat_name] = {
                    "orientation": orientation,
                    "mean_spacing": mean_spacing,
                    "min_spacing": min_spacing,
                    "max_spacing": max_spacing,
                    "standard_deviation": std_dev,
                    "coefficient_of_variation": cv,
                    "closest_standard": closest_standard if self.reference_scale.get_unit() == "inches" else
                                        self.reference_scale.convert_units(closest_standard, "inches",
                                                                         self.reference_scale.get_unit()),
                    "is_standard": is_standard,
                    "unit": self.reference_scale.get_unit(),
                    "confidence": spacing_confidence,
                    "center_points": centers,
                    "spacings": spacings_real
                }

                # Add clear spacings if available
                if clear_spacings_real is not None:
                    results[cat_name]["clear_spacings"] = clear_spacings_real
                    results[cat_name]["mean_clear_spacing"] = sum(clear_spacings_real) / len(clear_spacings_real)

        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            "status": "success",
            "spacings": results,
            "confidence": overall_confidence
        }

    def find_standard_spacing(
        self,
        detections: List[Dict],
        category_name: Optional[str] = None
    ) -> Dict:
        """
        Identify if framing members follow standard spacing.

        Args:
            detections: List of detection dictionaries
            category_name: Focus on specific category (e.g., "stud", "joist")

        Returns:
            Dict: Standard spacing analysis
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        # Get spacing data
        spacing_data = self.calculate_spacings(detections)

        if spacing_data["status"] != "success":
            return spacing_data

        spacings = spacing_data["spacings"]

        # Filter by category if specified
        if category_name is not None:
            if category_name not in spacings:
                return {
                    "status": "error",
                    "message": f"Category '{category_name}' not found or insufficient members"
                }

            categories_to_check = {category_name: spacings[category_name]}
        else:
            categories_to_check = spacings

        # Check each category for standard spacing
        standard_results = {}

        for cat, data in categories_to_check.items():
            if "is_standard" in data and data["is_standard"]:
                standard = data["closest_standard"]
                standard_results[cat] = {
                    "is_standard": True,
                    "standard_spacing": standard,
                    "actual_mean": data["mean_spacing"],
                    "confidence": data["confidence"],
                    "unit": data["unit"]
                }
            else:
                standard_results[cat] = {
                    "is_standard": False,
                    "closest_standard": data.get("closest_standard"),
                    "actual_mean": data.get("mean_spacing"),
                    "confidence": data.get("confidence", 0),
                    "unit": data.get("unit")
                }

        return {
            "status": "success",
            "standard_spacing_analysis": standard_results
        }


# models/measurements/dimension_estimator.py
"""
Module for estimating dimensions of framing members.
"""



# models/measurements/path_calculator.py
"""
Module for calculating wiring paths and distances.
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("path_calculator")

class PathCalculator:
    """
    Class for calculating wiring paths and distances.
    """

    def __init__(self, reference_scale: ReferenceScale):
        """
        Initialize the path calculator.

        Args:
            reference_scale: Reference scale for conversions
        """
        self.reference_scale = reference_scale

    def calculate_path(
        self,
        path_points: List[Tuple[int, int]],
        image: Optional[np.ndarray] = None,
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Calculate the total distance of a wiring path.

        Args:
            path_points: List of (x, y) points defining the path
            image: Original image (optional)
            drill_points: List of points where drilling is required

        Returns:
            Dict: Path analysis results
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        if len(path_points) < 2:
            return {
                "status": "error",
                "message": "At least two points needed for a path",
            }

        # Calculate path segments and distances
        segments = []
        total_pixel_distance = 0
        total_real_distance = 0

        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i+1]

            # Calculate segment distance in pixels
            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Convert to real distance
            real_distance = self.reference_scale.pixels_to_real_distance(pixel_distance)

            # Update totals
            total_pixel_distance += pixel_distance
            total_real_distance += real_distance

            # Store segment information
            segments.append({
                "start": (x1, y1),
                "end": (x2, y2),
                "pixel_distance": pixel_distance,
                "real_distance": real_distance,
                "unit": self.reference_scale.get_unit()
            })

        # Process drill points
        processed_drill_points = []

        if drill_points:
            for point in drill_points:
                processed_drill_points.append({
                    "position": point,
                    "requires_drilling": True
                })

        # Round values for cleaner output
        total_real_distance_rounded = round(total_real_distance, 2)

        # Convert to feet if in inches and distance is large
        display_distance = total_real_distance_rounded
        display_unit = self.reference_scale.get_unit()

        if display_unit == "inches" and total_real_distance_rounded > 24:
            display_distance = total_real_distance_rounded / 12
            display_unit = "feet"

        # Return results
        return {
            "status": "success",
            "path_segments": segments,
            "total_distance": total_real_distance_rounded,
            "display_distance": round(display_distance, 2),
            "display_unit": display_unit,
            "unit": self.reference_scale.get_unit(),
            "drill_points": processed_drill_points,
            "drill_count": len(processed_drill_points)
        }

    def identify_drill_points(
        self,
        path_points: List[Tuple[int, int]],
        detections: List[Dict],
        image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Identify points where the path intersects with framing members.

        Args:
            path_points: List of (x, y) points defining the path
            detections: List of detection dictionaries
            image: Original image (optional)

        Returns:
            List[Dict]: List of drill points
        """
        if len(path_points) < 2:
            return []

        drill_points = []

        # Process each path segment
        for i in range(len(path_points) - 1):
            start_x, start_y = path_points[i]
            end_x, end_y = path_points[i+1]

            # Check intersection with each framing member
            for det in detections:
                # Skip non-framing categories
                category = det['category_name']
                if category not in ['stud', 'joist', 'rafter', 'beam', 'plate', 'header']:
                    continue

                # Get bounding box
                bbox = det['bbox']
                x, y, w, h = bbox

                # Check if segment intersects the bounding box
                if self._segment_intersects_box(start_x, start_y, end_x, end_y, x, y, w, h):
                    # Calculate intersection point
                    intersection = self._get_segment_box_intersection(
                        start_x, start_y, end_x, end_y, x, y, w, h
                    )

                    if intersection:
                        # Determine drill difficulty based on member type and size
                        difficulty = self._calculate_drill_difficulty(det)

                        drill_points.append({
                            "position": intersection,
                            "requires_drilling": True,
                            "category": category,
                            "difficulty": difficulty
                        })

        return drill_points

    def _segment_intersects_box(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        box_x: float,
        box_y: float,
        box_width: float,
        box_height: float
    ) -> bool:
        """
        Check if a line segment intersects with a bounding box.

        Args:
            start_x, start_y: Start point of segment
            end_x, end_y: End point of segment
            box_x, box_y, box_width, box_height: Bounding box

        Returns:
            bool: True if the segment intersects the box
        """
        # Define box corners
        left = box_x
        right = box_x + box_width
        top = box_y
        bottom = box_y + box_height

        # Check if either endpoint is inside the box
        if (left <= start_x <= right and top <= start_y <= bottom) or \
           (left <= end_x <= right and top <= end_y <= bottom):
            return True

        # Check if line segment intersects any of the box edges
        edges = [
            (left, top, right, top),      # Top edge
            (right, top, right, bottom),    # Right edge
            (left, bottom, right, bottom),   # Bottom edge
            (left, top, left, bottom)     # Left edge
        ]

        for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
            if self._line_segments_intersect(
                start_x, start_y, end_x, end_y,
                edge_x1, edge_y1, edge_x2, edge_y2
            ):
                return True

        return False

    def _line_segments_intersect(
        self,
        a_x1: float, a_y1: float, a_x2: float, a_y2: float,
        b_x1: float, b_y1: float, b_x2: float, b_y2: float
    ) -> bool:
        """
        Check if two line segments intersect.

        Args:
            a_x1, a_y1, a_x2, a_y2: First line segment
            b_x1, b_y1, b_x2, b_y2: Second line segment

        Returns:
            bool: True if the segments intersect
        """
        # Calculate the direction vectors
        r = (a_x2 - a_x1, a_y2 - a_y1)
        s = (b_x2 - b_x1, b_y2 - b_y1)

        # Calculate the cross product (r × s)
        rxs = r[0] * s[1] - r[1] * s[0]

        # If r × s = 0, the lines are collinear or parallel
        if abs(rxs) < 1e-8:
            return False

        # Calculate t and u parameters
        q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

        # Check if intersection point is within both segments
        return 0 <= t <= 1 and 0 <= u <= 1

    def _get_segment_box_intersection(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        box_x: float,
        box_y: float,
        box_width: float,
        box_height: float
    ) -> Optional[Tuple[float, float]]:
        """
        Get the intersection point of a line segment with a box.

        Args:
            start_x, start_y: Start point of segment
            end_x, end_y: End point of segment
            box_x, box_y, box_width, box_height: Bounding box

        Returns:
            Optional[Tuple[float, float]]: Intersection point or None
        """
        # Define box corners
        left = box_x
        right = box_x + box_width
        top = box_y
        bottom = box_y + box_height

        # If start point is inside the box, use it
        if left <= start_x <= right and top <= start_y <= bottom:
            return (start_x, start_y)

        # If end point is inside the box, use it
        if left <= end_x <= right and top <= end_y <= bottom:
            return (end_x, end_y)

        # Check intersections with box edges
        edges = [
            (left, top, right, top),      # Top edge
            (right, top, right, bottom),    # Right edge
            (left, bottom, right, bottom),   # Bottom edge
            (left, top, left, bottom)     # Left edge
        ]

        for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
            if self._line_segments_intersect(
                start_x, start_y, end_x, end_y,
                edge_x1, edge_y1, edge_x2, edge_y2
            ):
                # Calculate intersection point
                intersection = self._calculate_intersection_point(
                    start_x, start_y, end_x, end_y,
                    edge_x1, edge_y1, edge_x2, edge_y2
                )

                if intersection:
                    return intersection

        return None

    def _calculate_intersection_point(
        self,
        a_x1: float, a_y1: float, a_x2: float, a_y2: float,
        b_x1: float, b_y1: float, b_x2: float, b_y2: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate the intersection point of two line segments.

        Args:
            a_x1, a_y1, a_x2, a_y2: First line segment
            b_x1, b_y1, b_x2, b_y2: Second line segment

        Returns:
            Optional[Tuple[float, float]]: Intersection point or None
        """
        # Calculate the direction vectors
        r = (a_x2 - a_x1, a_y2 - a_y1)
        s = (b_x2 - b_x1, b_y2 - b_y1)

        # Calculate the cross product (r × s)
        rxs = r[0] * s[1] - r[1] * s[0]

        # If r × s = 0, the lines are collinear or parallel
        if abs(rxs) < 1e-8:
            return None

        # Calculate t parameter
        q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

        # Check if intersection point is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix = a_x1 + t * r[0]
            iy = a_y1 + t * r[1]
            return (ix, iy)

        return None

    def _calculate_drill_difficulty(self, detection: Dict) -> str:
        """
        Estimate the difficulty of drilling through a detected member.

        Args:
            detection: Detection dictionary

        Returns:
            str: Difficulty level ("easy", "moderate", "difficult")
        """
        category = detection['category_name']

        # Estimate based on category
        if category in ['stud', 'plate']:
            base_difficulty = "easy"
        elif category in ['joist', 'rafter']:
            base_difficulty = "moderate"
        elif category in ['beam', 'header']:
            base_difficulty = "difficult"
        else:
            base_difficulty = "moderate"

        # Adjust based on member size if available
        if 'dimensions' in detection:
            thickness = detection['dimensions'].get('thickness', 0)

            # Convert to inches for standard comparison
            if self.reference_scale.get_unit() != "inches":
                thickness_inches = self.reference_scale.convert_units(
                    thickness, self.reference_scale.get_unit(), "inches"
                )
            else:
                thickness_inches = thickness

            # Adjust difficulty based on thickness
            if thickness_inches > 3.0: # Thick member
                if base_difficulty == "easy":
                    base_difficulty = "moderate"
                elif base_difficulty == "moderate":
                    base_difficulty = "difficult"
            elif thickness_inches < 1.0: # Thin member
                if base_difficulty == "difficult":
                    base_difficulty = "moderate"
                elif base_difficulty == "moderate":
                    base_difficulty = "easy"

        return base_difficulty

    def estimate_drilling_time(
        self,
        drill_points: List[Dict],
        drill_speed: str = "normal"
    ) -> Dict:
        """
        Estimate the time required for drilling through framing members.

        Args:
            drill_points: List of drill points
            drill_speed: Drilling speed ("slow", "normal", "fast")

        Returns:
            Dict: Time estimates
        """
        if not drill_points:
            return {
                "total_time_minutes": 0,
                "drill_points": 0,
                "average_time_per_point": 0
            }

        # Base time per difficulty level (in minutes)
        time_factors = {
            "easy": {"slow": 5, "normal": 3, "fast": 2},
            "moderate": {"slow": 8, "normal": 5, "fast": 3},
            "difficult": {"slow": 12, "normal": 8, "fast": 5}
        }

        total_time = 0

        for point in drill_points:
            difficulty = point.get("difficulty", "moderate")
            time_per_drill = time_factors.get(difficulty, time_factors["moderate"])[drill_speed]
            total_time += time_per_drill

        return {
            "total_time_minutes": total_time,
            "drill_points": len(drill_points),
            "average_time_per_point": total_time / len(drill_points)
        }


# models/measurements/visualization.py
"""
Visualization utilities for measurement estimations.
"""

import numpy as np
import cv2
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger

logger = get_logger("visualization")

def visualize_measurements(
    image: np.ndarray,
    measurement_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_dimensions: bool = True,
    show_spacings: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize measurement results on an image.

    Args:
        image: Input image
        measurement_result: Measurement analysis results
        output_path: Path to save visualization
        show_dimensions: Whether to show member dimensions
        show_spacings: Whether to show spacing measurements
        line_thickness: Line thickness for drawings
        font_scale: Font scale for text

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # White

    # Get measurements data
    if 'measurements' not in measurement_result:
        logger.warning("No measurements data in result")
        return vis_img

    measurements = measurement_result['measurements']
    unit = measurements.get('unit', 'inches')

    # Visualize dimensions for each member
    if show_dimensions and 'dimensions' in measurements:
        for cat_name, dim_data in measurements['dimensions'].items():
            # Set color based on category
            if cat_name == 'stud':
                color = (0, 255, 0)  # Green
            elif cat_name == 'joist':
                color = (255, 0, 0)  # Blue
            elif cat_name == 'rafter':
                color = (0, 0, 255) # Red
            elif cat_name == 'beam':
                color = (0, 255, 255) # Yellow
            else:
                color = (255, 255, 0) # Cyan

            # Draw dimensions for each member
            for member in dim_data.get('dimensions', []):
                bbox = member.get('bbox')
                if not bbox:
                    continue

                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), color, line_thickness)

                # Draw dimension labels
                size_name = member.get('standard_size', 'unknown')
                thickness = round(member.get('thickness', 0), 1)
                depth = round(member.get('depth', 0), 1)

                if size_name != 'custom':
                    label = f"{size_name} ({thickness} x {depth} {unit})"
                else:
                    label = f"{thickness} x {depth} {unit}"

                # Draw label background
                text_size = cv2.getTextSize(label, font, font_scale, line_thickness)[0]
                cv2.rectangle(
                    vis_img,
                    (int(x), int(y - text_size[1] - 5)),
                    (int(x + text_size[0]), int(y)),
                    color,
                    -1
                )

                # Draw label
                cv2.putText(
                    vis_img,
                    label,
                    (int(x), int(y - 5)),
                    font,
                    font_scale,
                    text_color,
                    1
                )

    # Visualize spacings between members
    if show_spacings and 'spacings' in measurements:
        for cat_name, spacing_data in measurements['spacings'].items():
            # Set color based on category
            if cat_name == 'stud':
                color = (0, 165, 255) # Orange
            elif cat_name == 'joist':
                color = (255, 0, 255) # Magenta
            elif cat_name == 'rafter':
                color = (128, 0, 128) # Purple
            else:
                color = (128, 128, 0) # Olive

            # Get center points and spacings
            center_points = spacing_data.get('center_points', [])
            spacings = spacing_data.get('spacings', [])
            orientation = spacing_data.get('orientation', 'mixed')

            # Draw lines between centers
            if len(center_points) < 2:
                continue

            for i in range(len(center_points) - 1):
                p1 = (int(center_points[i][0]), int(center_points[i][1]))
                p2 = (int(center_points[i+1][0]), int(center_points[i+1][1]))

                # Draw line between centers
                cv2.line(vis_img, p1, p2, color, line_thickness)

                # Draw spacing measurement
                if i < len(spacings):
                    spacing_value = round(spacings[i], 1)
                    spacing_label = f"{spacing_value} {unit}"

                    # Calculate label position
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2

                    # Adjust position based on orientation
                    if orientation == "vertical":
                        text_pos = (mid_x, mid_y - 10)
                    elif orientation == "horizontal":
                        text_pos = (mid_x + 10, mid_y)
                    else:
                        text_pos = (mid_x, mid_y - 10)

                    # Draw label background
                    text_size = cv2.getTextSize(spacing_label, font, font_scale, 1)[0]
                    cv2.rectangle(
                        vis_img,
                        (int(text_pos[0]), int(text_pos[1] - text_size[1])),
                        (int(text_pos[0] + text_size[0]), int(text_pos[1] + 5)),
                        color,
                        -1
                    )

                    # Draw label
                    cv2.putText(
                        vis_img,
                        spacing_label,
                        text_pos,
                        font,
                        font_scale,
                        text_color,
                        1
                    )

    # Add overall information
    confidence = measurement_result.get('confidence', 0)
    confidence_text = f"Confidence: {confidence:.2f}"

    cv2.putText(
        vis_img,
        confidence_text,
        (10, 30),
        font,
        0.7,
        (0, 255, 255),
        2
    )

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved measurement visualization to {output_path}")

    return vis_img

def visualize_wiring_path(
    image: np.ndarray,
    path_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_distance: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Visualize a wiring path on an image.

    Args:
        image: Input image
        path_result: Path analysis results
        output_path: Path to save visualization
        show_distance: Whether to show distance measurements
        line_thickness: Line thickness for drawings

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    path_color = (0, 255, 255)  # Yellow
    drill_color = (0, 0, 255)   # Red
    text_color = (255, 255, 255)  # White

    # Draw path segments
    segments = path_result.get('path_segments', [])
    for segment in segments:
        start = segment.get('start')
        end = segment.get('end')

        if not start or not end:
            continue

        # Draw line segment
        cv2.line(
            vis_img,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            path_color,
            line_thickness
        )

        # Draw distance if requested
        if show_distance:
            real_distance = segment.get('real_distance', 0)
            unit = segment.get('unit', 'inches')
            distance_label = f"{real_distance:.1f} {unit}"

            # Calculate label position
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2

            # Draw label background
            text_size = cv2.getTextSize(distance_label, font, 0.5, 1)[0]
            cv2.rectangle(
                vis_img,
                (int(mid_x), int(mid_y - text_size[1])),
                (int(mid_x + text_size[0]), int(mid_y + 5)),
                path_color,
                -1
            )

            # Draw label
            cv2.putText(
                vis_img,
                distance_label,
                (int(mid_x), int(mid_y)),
                font,
                0.5,
                text_color,
                1
            )

    # Draw drill points
    drill_points = path_result.get('drill_points', [])
    for point in drill_points:
        position = point.get('position')
        difficulty = point.get('difficulty', 'moderate')

        if not position:
            continue

        # Adjust color based on difficulty
        if difficulty == 'easy':
            point_color = (0, 255, 0) # Green
        elif difficulty == 'moderate':
            point_color = (0, 165, 255) # Orange
        else: # difficult
            point_color = (0, 0, 255)   # Red

        # Draw drill point
        cv2.circle(
            vis_img,
            (int(position[0]), int(position[1])),
            8,
            point_color,
            -1
        )

        # Draw label
        cv2.putText(
            vis_img,
            difficulty,
            (int(position[0]) + 10, int(position[1])),
            font,
            0.5,
            point_color,
            2
        )

    # Add total distance at the top
    total_distance = path_result.get('display_distance', 0)
    display_unit = path_result.get('display_unit', 'inches')
    drill_count = path_result.get('drill_count', 0)

    total_text = f"Total: {total_distance:.1f} {display_unit}"
    drill_text = f"Drill points: {drill_count}"

    cv2.putText(vis_img, total_text, (10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_img, drill_text, (10, 60), font, 0.7, (0, 0, 255), 2)

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved wiring path visualization to {output_path}")

    return vis_img

def visualize_scale_calibration(
    image: np.ndarray,
    calibration_data: Dict,
    output_path: Optional[Union[str, Path]] = None,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Visualize scale calibration on an image.

    Args:
        image: Input image
        calibration_data: Calibration data
        output_path: Path to save visualization
        line_thickness: Line thickness for drawings

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    calib_color = (0, 255, 0)  # Green
    text_color = (255, 255, 255)  # White

    method = calibration_data.get('method', 'unknown')

    # Draw calibration visualization based on method
    if method == 'point_distance':
        # Draw reference line between points
        reference_data = calibration_data.get('reference_data', {})
        points = reference_data.get('points', [])

        if len(points) >= 2:
            p1 = (int(points[0][0]), int(points[0][1]))
            p2 = (int(points[1][0]), int(points[1][1]))

            # Draw reference line
            cv2.line(vis_img, p1, p2, calib_color, line_thickness)

            # Draw endpoints
            cv2.circle(vis_img, p1, 5, calib_color, -1)
            cv2.circle(vis_img, p2, 5, calib_color, -1)

            # Draw distance label
            known_distance = reference_data.get('known_distance', 0)
            unit = calibration_data.get('unit', 'inches')
            label = f"{known_distance} {unit}"

            # Calculate label position
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            # Draw label background
            text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
            cv2.rectangle(
                vis_img,
                (mid_x, mid_y - text_size[1] - 5),
                (mid_x + text_size[0], mid_y),
                calib_color,
                -1
            )

            # Draw label
            cv2.putText(
                vis_img,
                label,
                (mid_x, mid_y - 5),
                font,
                0.6,
                text_color,
                1
            )

    elif method == 'object_dimensions':
        # Draw reference object box
        reference_data = calibration_data.get('reference_data', {})
        object_bbox = reference_data.get('object_bbox', [])

        if len(object_bbox) == 4:
            x, y, w, h = object_bbox

            # Draw bounding box
            cv2.rectangle(
                vis_img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                calib_color,
                line_thickness
            )

            # Draw dimension labels
            dimensions = reference_data.get('object_dimensions', [0, 0])
            unit = calibration_data.get('unit', 'inches')

            width_label = f"Width: {dimensions[0]} {unit}"
            height_label = f"Height: {dimensions[1]} {unit}"

            # Draw width label
            cv2.putText(
                vis_img,
                width_label,
                (int(x), int(y - 10)),
                font,
                0.6,
                calib_color,
                2
            )

            # Draw height label
            cv2.putText(
                vis_img,
                height_label,
                (int(x + w + 5), int(y + h // 2)),
                font,
                0.6,
                calib_color,
                2
            )

    # Add calibration information
    pixels_per_unit = calibration_data.get('pixels_per_unit', 0)
    unit = calibration_data.get('unit', 'inches')
    confidence = calibration_data.get('confidence', 0)

    info_text1 = f"Scale: {pixels_per_unit:.2f} pixels per {unit}"
    info_text2 = f"Confidence: {confidence:.2f}"

    cv2.putText(vis_img, info_text1, (10, 30), font, 0.7, calib_color, 2)
    cv2.putText(vis_img, info_text2, (10, 60), font, 0.7, calib_color, 2)

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved calibration visualization to {output_path}")

    return vis_img








#Now I'll implement the end-to-end integration code to connect all the components of the electrician time estimation application together. This will involve creating code that:

#1. Connects the frontend UI to the backend API
#2. Ensures proper data flow between detection, estimation, and display components
#3. Implements error handling and recovery
#4. Adds progress indicators for long-running processes
#5. Creates a configuration system for application settings

#Let me start by implementing the backend API routes that will serve the frontend:


# api/routes/estimation.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import tempfile
import json
import os
import shutil
from typing import Dict, List, Optional, Any

from models.detection import FramingDetector, detect_framing
from models.measurements import MeasurementEstimator
from utils.logger import get_logger
from utils.exceptions import EstimatorBaseException
from config.settings import (
    MODEL_WEIGHTS_DIR, IMAGES_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    COMPLEXITY_THRESHOLDS, BASE_TIME_MINUTES
)

logger = get_logger("api_estimation")
router = APIRouter()

# Task storage for background tasks
TASKS = {}

# Initialize models
detector = None
estimator = None

def load_models():
    """
    Lazy-load models when needed.
    """
    global detector, estimator
    
    if detector is None:
        # Find best available model weights
        model_paths = list(MODEL_WEIGHTS_DIR.glob("framing_detector_*.pt"))
        if model_paths:
            model_path = sorted(model_paths)[-1]  # Use most recent model
            logger.info(f"Loading detection model from {model_path}")
            detector = FramingDetector.from_checkpoint(model_path)
        else:
            logger.warning("No detection model weights found, using pretrained model")
            detector = FramingDetector(pretrained=True)
    
    if estimator is None:
        estimator = MeasurementEstimator()
    
    return detector, estimator

def process_image_task(task_id: str, image_path: Path):
    """
    Background task to process an image.
    """
    try:
        # Update task status
        TASKS[task_id]["status"] = "processing"
        
        # Load the models
        detector, estimator = load_models()
        
        # Run detection
        detection_result = detect_framing(detector, str(image_path))
        TASKS[task_id]["progress"] = 40
        
        # If we have a stored calibration, use it
        calibration_data = TASKS[task_id].get("calibration_data")
        if calibration_data:
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        else:
            # Try to auto-calibrate using a known framing member
            if detection_result["detections"]:
                # Find a likely stud or joist for calibration
                for det in detection_result["detections"]:
                    if det["category_name"] in ["stud", "joist"] and det["confidence"] > 0.8:
                        # Use standard framing dimensions for calibration
                        # 2x4 stud is typically 1.5 x 3.5 inches
                        try:
                            estimator.calibrate_from_known_object(
                                detection_result["image"],
                                det["bbox"],
                                (1.5, 3.5)  # 2x4 nominal dimensions
                            )
                            TASKS[task_id]["calibration_data"] = estimator.reference_scale.get_calibration_data()
                            break
                        except Exception as e:
                            logger.warning(f"Auto-calibration failed: {str(e)}")
        
        TASKS[task_id]["progress"] = 60
        
        # Run measurements
        measurement_result = estimator.analyze_framing_measurements(
            detection_result, calibration_check=False
        )
        TASKS[task_id]["progress"] = 80
        
        # Estimate time required based on complexity
        time_estimate = estimate_time(detection_result, measurement_result)
        
        # Save results
        results = {
            "detections": detection_result["detections"],
            "measurements": measurement_result["measurements"],
            "time_estimate": time_estimate
        }
        
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["results"] = results
        
        # Clean up - keep for 1 hour in a real app
        # del TASKS[task_id]
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)

def estimate_time(detection_result: Dict, measurement_result: Dict) -> Dict:
    """
    Estimate time required for electrical work based on detection and measurements.
    """
    # Count different framing members
    member_counts = {}
    for det in detection_result["detections"]:
        cat = det["category_name"]
        member_counts[cat] = member_counts.get(cat, 0) + 1
    
    # Calculate complexity factors
    stud_count = member_counts.get("stud", 0)
    joist_count = member_counts.get("joist", 0)
    obstacle_count = member_counts.get("obstacle", 0) + member_counts.get("plumbing", 0)
    electrical_box_count = member_counts.get("electrical_box", 0)
    
    # Determine spacing complexity (irregular spacing is more complex)
    spacing_complexity = 0
    spacings = measurement_result.get("measurements", {}).get("spacing", {}).get("spacings", {})
    
    for cat, spacing_data in spacings.items():
        if not spacing_data.get("is_standard", True):
            spacing_complexity += 0.3
    
    # Calculate overall complexity score
    complexity_score = (
        stud_count * 0.05 +
        joist_count * 0.05 +
        obstacle_count * 0.2 +
        electrical_box_count * 0.1 +
        spacing_complexity
    )
    
    # Map to complexity level
    if complexity_score <= COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"
    elif complexity_score <= COMPLEXITY_THRESHOLDS["moderate"]:
        complexity = "moderate"
    else:
        complexity = "complex"
    
    # Calculate time estimate
    base_time = BASE_TIME_MINUTES[complexity]
    total_time = base_time + (stud_count * 0.5) + (obstacle_count * 1.5) + (electrical_box_count * 2)
    
    return {
        "complexity": complexity,
        "complexity_score": complexity_score,
        "estimated_minutes": round(total_time),
        "factors": {
            "stud_count": stud_count,
            "joist_count": joist_count,
            "obstacle_count": obstacle_count,
            "electrical_box_count": electrical_box_count,
            "spacing_complexity": spacing_complexity
        }
    }

@router.post("/estimate")
async def estimate_from_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    calibration: Optional[str] = Form(None)
):
    """
    Submit an image for time estimation (processes in background).
    Returns a task ID for checking status.
    """
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Create temp directory for this task
    task_dir = IMAGES_DIR / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    image_path = task_dir / f"input{Path(file.filename).suffix}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse calibration data if provided
    calibration_data = None
    if calibration:
        try:
            calibration_data = json.loads(calibration)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid calibration data format")
    
    # Create task entry
    TASKS[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "image_path": str(image_path),
        "calibration_data": calibration_data
    }
    
    # Start background processing
    background_tasks.add_task(process_image_task, task_id, image_path)
    
    return {"task_id": task_id, "status": "pending"}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a submitted task.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id].copy()
    
    # Don't return the image path in the response
    if "image_path" in task:
        del task["image_path"]
    
    return task

@router.post("/calibrate")
async def calibrate_from_image(
    file: UploadFile = File(...),
    reference_type: str = Form(...),
    reference_data: str = Form(...)
):
    """
    Calibrate the measurement system using a reference object or points.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Load the image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Initialize the estimator
        _, estimator = load_models()
        
        # Parse reference data
        ref_data = json.loads(reference_data)
        
        # Perform calibration based on reference type
        if reference_type == "points":
            if "points" not in ref_data or "distance" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing points or distance")
            
            points = ref_data["points"]
            distance = float(ref_data["distance"])
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_reference(
                image, points, distance, units
            )
        
        elif reference_type == "object":
            if "bbox" not in ref_data or "dimensions" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing bbox or dimensions")
            
            bbox = ref_data["bbox"]
            dimensions = ref_data["dimensions"]
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_known_object(
                image, bbox, dimensions, units
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid reference type")
        
        # Clean up
        os.unlink(temp_path)
        
        return calibration
        
    except EstimatorBaseException as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        logger.error(f"Calibration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during calibration")

@router.post("/path")
async def estimate_wiring_path(
    file: UploadFile = File(...),
    path_points: str = Form(...),
    calibration: Optional[str] = Form(None)
):
    """
    Estimate wiring path distances and drilling points.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Load the image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Load models
        detector, estimator = load_models()
        
        # Parse path points
        points = json.loads(path_points)
        
        # Parse calibration data if provided
        if calibration:
            calibration_data = json.loads(calibration)
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        
        # Run detection
        detection_result = detect_framing(detector, image)
        
        # Estimate wiring path
        path_result = estimator.estimate_wiring_path(
            detection_result=detection_result,
            path_points=points
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return path_result
        
    except EstimatorBaseException as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        logger.error(f"Path estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during path estimation")


#Now, let's implement the frontend part with Electron. I'll start with the main Electron app file:

javascript
// frontend/electron/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const url = require('url');

// Keep a global reference of the window object
let mainWindow;

// Configuration
const config = {
  apiBaseUrl: 'http://localhost:8000/api',
  settings: {
    autoCalibrate: true,
    confidenceThreshold: 0.25,
    showMeasurements: true
  }
};

// Load config from file if exists
try {
  const userDataPath = app.getPath('userData');
  const configPath = path.join(userDataPath, 'config.json');
  if (fs.existsSync(configPath)) {
    const savedConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    config.settings = {...config.settings, ...savedConfig.settings};
    if (savedConfig.apiBaseUrl) {
      config.apiBaseUrl = savedConfig.apiBaseUrl;
    }
  }
} catch (error) {
  console.error('Error loading config:', error);
}

// Save config to file
function saveConfig() {
  try {
    const userDataPath = app.getPath('userData');
    const configPath = path.join(userDataPath, 'config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  } catch (error) {
    console.error('Error saving config:', error);
  }
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true, // Be careful with this in production!
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Load the index.html of the app
  mainWindow.loadFile(path.join(__dirname, 'src/index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed
  mainWindow.on('closed', function() {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed
app.on('window-all-closed', function() {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function() {
  if (mainWindow === null) createWindow();
});

// IPC handlers
ipcMain.handle('select-image', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }]
  });
  
  if (canceled) return null;
  return filePaths[0];
});

ipcMain.handle('get-config', () => {
  return config;
});

ipcMain.handle('save-config', (event, newSettings) => {
  config.settings = {...config.settings, ...newSettings};
  config.apiBaseUrl = newSettings.apiBaseUrl || config.apiBaseUrl;
  saveConfig();
  return config;
});

ipcMain.handle('submit-image', async (event, imagePath, calibrationData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    
    if (calibrationData) {
      formData.append('calibration', JSON.stringify(calibrationData));
    }
    
    const response = await axios.post(
      `${config.apiBaseUrl}/estimate`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000 // 30 seconds timeout
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error submitting image:', error);
    throw new Error(error.response?.data?.detail || 'Failed to submit image');
  }
});

ipcMain.handle('check-task', async (event, taskId) => {
  try {
    const response = await axios.get(
      `${config.apiBaseUrl}/tasks/${taskId}`,
      { timeout: 10000 }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error checking task:', error);
    throw new Error(error.response?.data?.detail || 'Failed to check task status');
  }
});

ipcMain.handle('calibrate-image', async (event, imagePath, referenceType, referenceData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    formData.append('reference_type', referenceType);
    formData.append('reference_data', JSON.stringify(referenceData));
    
    const response = await axios.post(
      `${config.apiBaseUrl}/calibrate`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error calibrating image:', error);
    throw new Error(error.response?.data?.detail || 'Failed to calibrate image');
  }
});

ipcMain.handle('estimate-path', async (event, imagePath, pathPoints, calibrationData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    formData.append('path_points', JSON.stringify(pathPoints));
    
    if (calibrationData) {
      formData.append('calibration', JSON.stringify(calibrationData));
    }
    
    const response = await axios.post(
      `${config.apiBaseUrl}/path`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error estimating path:', error);
    throw new Error(error.response?.data?.detail || 'Failed to estimate wiring path');
  }
});


#Now, let's define a preload script that will expose the IPC functions to our renderer process:

javascript
// frontend/electron/preload.js
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    selectImage: () => ipcRenderer.invoke('select-image'),
    getConfig: () => ipcRenderer.invoke('get-config'),
    saveConfig: (settings) => ipcRenderer.invoke('save-config', settings),
    submitImage: (imagePath, calibrationData) => ipcRenderer.invoke('submit-image', imagePath, calibrationData),
    checkTask: (taskId) => ipcRenderer.invoke('check-task', taskId),
    calibrateImage: (imagePath, referenceType, referenceData) => 
      ipcRenderer.invoke('calibrate-image', imagePath, referenceType, referenceData),
    estimatePath: (imagePath, pathPoints, calibrationData) => 
      ipcRenderer.invoke('estimate-path', imagePath, pathPoints, calibrationData)
  }
);


#Next, let's create the main HTML file for the frontend:

html
<!-- frontend/electron/src/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electrician Time Estimator</title>
    <link rel="stylesheet" href="styles/main.css">
</head>
<body>
    <header>
        <h1>Electrician Time Estimator</h1>
        <div class="toolbar">
            <button id="selectImageBtn">Select Image</button>
            <button id="calibrateBtn">Calibrate</button>
            <button id="drawPathBtn">Draw Wiring Path</button>
            <button id="settingsBtn">Settings</button>
        </div>
    </header>

    <main>
        <div id="imageContainer">
            <div id="placeholder">
                <p>Select an image to analyze</p>
            </div>
            <div id="canvasContainer" style="display: none;">
                <canvas id="mainCanvas"></canvas>
            </div>
            <div id="progress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <p class="progress-text">Processing...</p>
            </div>
        </div>

        <div id="resultsPanel">
            <div class="panel-header">
                <h2>Results</h2>
                <button id="exportBtn" disabled>Export Report</button>
            </div>
            <div id="resultsContent">
                <p class="placeholder">No results yet. Process an image to see time estimates.</p>
            </div>
        </div>
    </main>

    <div id="calibrateModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Calibrate Measurements</h2>
            <p>Select a calibration method:</p>
            
            <div class="calibration-options">
                <button class="calibrate-option" data-method="known-object">Use Known Object</button>
                <button class="calibrate-option" data-method="reference-points">Use Reference Points</button>
            </div>
            
            <div id="calibrateKnownObject" class="calibration-method">
                <p>Select a framing member by drawing a box around it:</p>
                <button id="drawBoxBtn">Draw Box</button>
                <div>
                    <label>Object Type:</label>
                    <select id="objectType">
                        <option value="2x4">2x4 Stud (1.5" x 3.5")</option>
                        <option value="2x6">2x6 (1.5" x 5.5")</option>
                        <option value="2x8">2x8 (1.5" x 7.25")</option>
                        <option value="4x4">4x4 Post (3.5" x 3.5")</option>
                        <option value="custom">Custom Dimensions</option>
                    </select>
                </div>
                <div id="customDimensions" style="display: none;">
                    <label>Width (inches): <input id="customWidth" type="number" step="0.25" min="0.5" value="1.5"></label>
                    <label>Height (inches): <input id="customHeight" type="number" step="0.25" min="0.5" value="3.5"></label>
                </div>
                <button id="submitKnownObjectBtn" disabled>Submit Calibration</button>
            </div>
            
            <div id="calibrateReferencePoints" class="calibration-method">
                <p>Click to place two points at a known distance:</p>
                <div>
                    <label>Distance between points:</label>
                    <input id="referenceDistance" type="number" step="0.25" min="0.5" value="16">
                    <select id="distanceUnit">
                        <option value="inches">inches</option>
                        <option value="feet">feet</option>
                        <option value="cm">centimeters</option>
                        <option value="m">meters</option>
                    </select>
                </div>
                <button id="submitPointsBtn" disabled>Submit Calibration</button>
            </div>
        </div>
    </div>

    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Settings</h2>
            
            <div class="settings-form">
                <div class="form-group">
                    <label for="apiUrl">API URL:</label>
                    <input type="text" id="apiUrl" placeholder="http://localhost:8000/api">
                </div>
                
                <div class="form-group">
                    <label for="confidenceThreshold">Detection Confidence Threshold:</label>
                    <input type="range" id="confidenceThreshold" min="0.05" max="0.95" step="0.05" value="0.25">
                    <span id="confidenceValue">0.25</span>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="autoCalibrate" checked>
                    <label for="autoCalibrate">Auto-calibrate using detected framing members</label>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="showMeasurements" checked>
                    <label for="showMeasurements">Show measurements on detection</label>
                </div>
                
                <button id="saveSettingsBtn">Save Settings</button>
            </div>
        </div>
    </div>

    <div id="pathModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Draw Wiring Path</h2>
            <p>Click on the image to place path points:</p>
            
            <div class="path-controls">
                <button id="clearPathBtn">Clear Path</button>
                <button id="undoPointBtn">Undo Last Point</button>
                <button id="calculatePathBtn" disabled>Calculate Path</button>
            </div>
            
            <div id="pathResults" style="display: none;">
                <h3>Wiring Path Results</h3>
                <div id="pathResultsContent"></div>
            </div>
        </div>
    </div>

    <script src="scripts/main.js"></script>
</body>
</html>


#Now, let's add some CSS styles:

css
/* frontend/electron/src/styles/main.css */
:root {
    --primary-color: #3498db;
<sourcefile>"""""
Goal : Coding an application that assisted with AI can function as an experienced electrician tasked with estimating the labor time for drilling through framing members to run electrical wiring in a residential home, by analyzing a photo, and following the request


____________________________________________________
#ddd;
this will be done in8;ple steps


* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}
step 1 Guide for coding
create logic behind the idea, how ai factors in, math behind the complexities

    color: var(--text-color);
    line-height: 1.6;
    background-color: #f9f9f9;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

header {
    background-color: var(--primary-color);
    color: white;
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 1.5rem;
}

main {
    display: flex;
    flex: 1;
    padding: 1rem;
    gap: 1rem;
    overflow: hidden;
}

button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

button:hover {
    background-color: var(--primary-dark);
}

button:disabled {
    background-color: var(--dark-gray);
    cursor: not-allowed;
}

.toolbar {
    display: flex;
    gap: 0.5rem;
}

#imageContainer {
    flex: 1.5;
    background-color: var(--light-gray);
    border-radius: 4px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
}

#placeholder {
    text-align: center;
    color: var(--dark-gray);
}

#canvasContainer {
    width: 100%;
    height: 100%;
    position: relative;
}

canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

#resultsPanel {
    flex: 1;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.panel-header {
    padding: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--light-gray);
    border-bottom: 1px solid var(--medium-gray);
}

.panel-header h2 {
    font-size: 1.2rem;
    font-weight: 500;
}

#resultsContent {
    padding: 1rem;
    overflow-y: auto;
    flex: 1;
}

.placeholder {
    color: var(--dark-gray);
    text-align: center;
    padding: 2rem 1rem;
}

/* Progress bar styles */
#progress {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-top: 1px solid var(--medium-gray);
}

.progress-bar {
    height: 10px;
    background-color: var(--light-gray);
    border-radius: 5px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--secondary-color);
    width: 0%;
    transition: width 0.3s ease-in-out;
}

.progress-text {
    text-align: center;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    z-index: 100;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background-color: white;
    margin: 10% auto;
    padding: 1.5rem;
    border-radius: 6px;
    width: 60%;
    max-width: 600px;
    position: relative;
}

.close-btn {
    position: absolute;
    top: 0.75rem;
    right: 1rem;
    font-size: 1.5rem;
    cursor: pointer;
}

.calibration-options {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.calibration-method {
    display: none;
    margin-top: 1rem;
    padding: 1rem;
    background-color: var(--light-gray);
    border-radius: 4px;
}

/* Form controls */
.form-group {
    margin-bottom: 1rem;
}

label {
    display: block;
    margin-bottom: 0.3rem;
}

input[type="text"],
input[type="number"],
select {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--medium-gray);
    border-radius: 4px;
}

input[type="range"] {
    width: 90%;
    vertical-align: middle;
}

.checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.checkbox label {
    margin-bottom: 0;
}

.checkbox input {
    width: auto;
}

/* Path drawing */
.path-controls {
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0;
}

/* Results styling */
.result-section {
    margin-bottom: 1.5rem;
}

.result-section h3 {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--light-gray);
}

.result-item {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
}

.detection-item {
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    background-color: var(--light-gray);
    border-radius: 4px;
}

.time-estimate {
    font-size: 1.5rem;
    text-align: center;
    padding: 1rem;
    margin: 1rem 0;
    background-color: var(--light-gray);
    border-radius: 4px;
    font-weight: bold;
}

.complexity-simple {
    color: var(--secondary-color);
}

.complexity-moderate {
    color: orange;
}

.complexity-complex {
    color: var(--accent-color);
}


#Finally, let's implement the main JavaScript file that will handle the UI and interactions:

javascript
// frontend/electron/src/scripts/main.js
// Global state
let state = {
  currentImage: null,
  imageElement: null,
  calibrationData: null,
  taskId: null,
  detectionResults: null,
  pathPoints: [],
  drawingBox: false,
  boxStart: null,
  referencePoints: [],
  isDrawingPath: false,
  settings: {
    apiBaseUrl: 'http://localhost:8000/api',
    confidenceThreshold: 0.25,
    autoCalibrate: true,
    showMeasurements: true
  }
};

// Canvas for image display and interaction
let canvas = document.getElementById('mainCanvas');
let ctx = canvas.getContext('2d');

// UI Elements
const selectImageBtn = document.getElementById('selectImageBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const drawPathBtn = document.getElementById('drawPathBtn');
const settingsBtn = document.getElementById('settingsBtn');
const exportBtn = document.getElementById('exportBtn');
const placeholder = document.getElementById('placeholder');
const canvasContainer = document.getElementById('canvasContainer');
const progressContainer = document.getElementById('progress');
const progressFill = document.querySelector('.progress-fill');
const progressText = document.querySelector('.progress-text');
const resultsContent = document.getElementById('resultsContent');

// Modal elements
const calibrateModal = document.getElementById('calibrateModal');
const settingsModal = document.getElementById('settingsModal');
const pathModal = document.getElementById('pathModal');
const closeButtons = document.querySelectorAll('.close-btn');

// Initialize the app
async function init() {
  // Load settings
  try {
    const config = await window.api.getConfig();
    state.settings = {...state.settings, ...config.settings};
    if (config.apiBaseUrl) {
      state.settings.apiBaseUrl = config.apiBaseUrl;
    }
    
    // Update UI with settings
    document.getElementById('apiUrl').value = state.settings.apiBaseUrl;
    document.getElementById('confidenceThreshold').value = state.settings.confidenceThreshold;
    document.getElementById('confidenceValue').textContent = state.settings.confidenceThreshold;
    document.getElementById('autoCalibrate').checked = state.settings.autoCalibrate;
    document.getElementById('showMeasurements').checked = state.settings.showMeasurements;
  } catch (error) {
    console.error('Error loading settings:', error);
  }
  
  // Set up event listeners
  setupEventListeners();
}

function setupEventListeners() {
  // Main buttons
  selectImageBtn.addEventListener('click', handleSelectImage);
  calibrateBtn.addEventListener('click', () => showModal(calibrateModal));
  drawPathBtn.addEventListener('click', startPathDrawing);
  settingsBtn.addEventListener('click', () => showModal(settingsModal));
  exportBtn.addEventListener('click', exportReport);
  
  // Close modal buttons
  closeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const modal = btn.closest('.modal');
      hideModal(modal);
    });
  });
  
  // Calibration modal
  document.querySelectorAll('.calibrate-option').forEach(btn => {
    btn.addEventListener('click', selectCalibrationMethod);
  });
  
  document.getElementById('objectType').addEventListener('change', handleObjectTypeChange);
  document.getElementById('drawBoxBtn').addEventListener('click', startBoxDrawing);
  document.getElementById('submitKnownObjectBtn').addEventListener('click', submitKnownObjectCalibration);
  document.getElementById('submitPointsBtn').addEventListener('click', submitPointsCalibration);
  
  // Settings modal
  document.getElementById('confidenceThreshold').addEventListener('input', updateConfidenceLabel);
  document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
  
  // Path modal
  document.getElementById('clearPathBtn').addEventListener('click', clearPath);
  document.getElementById('undoPointBtn').addEventListener('click', undoLastPathPoint);
  document.getElementById('calculatePathBtn').addEventListener('click', calculatePath);
  
  // Canvas interactions
  canvas.addEventListener('mousedown', handleCanvasMouseDown);
  canvas.addEventListener('mousemove', handleCanvasMouseMove);
  canvas.addEventListener('mouseup', handleCanvasMouseUp);
}

// Image selection and processing
async function handleSelectImage() {
  try {
    const imagePath = await window.api.selectImage();
    if (!imagePath) return;
    
    // Reset state
    state.currentImage = imagePath;
    state.detectionResults = null;
    state.taskId = null;
    exportBtn.disabled = true;
    
    // Show the image in canvas
    loadImageToCanvas(imagePath);
    
    // Hide placeholder, show canvas
    placeholder.style.display = 'none';
    canvasContainer.style.display = 'block';
    
    // Process the image
    await processImage();
  } catch (error) {
    showError('Failed to select image: ' + error.message);
  }
}

function loadImageToCanvas(imagePath) {
  // Create an image element
  const img = new Image();
  img.onload = function() {
    // Set canvas dimensions to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw image to canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    // Store image element
    state.imageElement = img;
  };
  
  // Set image source to file path (using URL scheme for Electron)
  img.src = 'file://' + imagePath;
}

async function processImage() {
  try {
    // Show progress bar
    showProgress(10, 'Submitting image...');
    
    // Submit the image for processing
    const taskResponse = await window.api.submitImage(
      state.currentImage, 
      state.calibrationData
    );
    
    state.taskId = taskResponse.task_id;
    
    // Poll for task completion
    await pollTaskStatus();
  } catch (error) {
    hideProgress();
    showError('Failed to process image: ' + error.message);
  }
}

async function pollTaskStatus() {
  if (!state.taskId) return;
  
  try {
    let completed = false;
    while (!completed) {
      const taskStatus = await window.api.checkTask(state.taskId);
      
      // Update progress
      showProgress(taskStatus.progress || 10, `${taskStatus.status}...`);
      
      if (taskStatus.status === 'completed') {
        // Process is complete
        state.detectionResults = taskStatus.results;
        
        // Display results
        displayResults(taskStatus.results);
        
        // Store calibration if available
        if (taskStatus.calibration_data) {
          state.calibrationData = taskStatus.calibration_data;
        }
        
        completed = true;
        hideProgress();
        exportBtn.disabled = false;
      } else if (taskStatus.status === 'error') {
        showError('Processing error: ' + taskStatus.error);
        hideProgress();
        completed = true;
      } else {
        // Wait a bit before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  } catch (error) {
    hideProgress();
    showError('Failed to check task status: ' + error.message);
  }
}

function displayResults(results) {
  if (!results) {
    resultsContent.innerHTML = '<p class="placeholder">No results available.</p>';
    return;
  }
  
  const { detections, measurements, time_estimate } = results;
  
  // Build the HTML for the results
  let html = '';
  
  // Time estimate section
  html += `
    <div class="result-section">
      <h3>Time Estimate</h3>
      <div class="time-estimate complexity-${time_estimate.complexity}">
        ${time_estimate.estimated_minutes} minutes
        <div style="font-size: 0.8rem; font-weight: normal;">
          Complexity: ${time_estimate.complexity}
        </div>
      </div>
    </div>
  `;
  
  // Detection stats
  html += `
    <div class="result-section">
      <h3>Detected Items</h3>
  `;
  
  // Group by category
  const categories = {};
  detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  // Add category counts
  for (const [category, count] of Object.entries(categories)) {
    html += `
      <div class="result-item">
        <span>${category}:</span>
        <span>${count}</span>
      </div>
    `;
  }
  
  html += `</div>`;
  
  // Measurements section
  if (measurements && measurements.dimensions) {
    html += `
      <div class="result-section">
        <h3>Framing Measurements</h3>
    `;
    
    const { dimensions } = measurements;
    
    for (const [category, data] of Object.entries(dimensions)) {
      if (data.member_count > 0) {
        html += `
          <div class="result-item">
            <span>${category}:</span>
            <span>${data.most_common_size}</span>
          </div>
        `;
      }
    }
    
    html += `</div>`;
  }
  
  // Detailed detections
  html += `
    <div class="result-section">
      <h3>Detection Details</h3>
      <div style="max-height: 200px; overflow-y: auto;">
  `;
  
  detections.forEach((det, index) => {
    html += `
      <div class="detection-item">
        <div><strong>Item ${index + 1}:</strong> ${det.category_name}</div>
        <div>Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
      </div>
    `;
  });
  
  html += `
      </div>
    </div>
  `;
  
  // Display the results
  resultsContent.innerHTML = html;
  
  // If measurement visualization is enabled, redraw with measurements
  if (state.settings.showMeasurements) {
    visualizeMeasurements(detections, measurements);
  } else {
    visualizeDetections(detections);
  }
}

function visualizeDetections(detections) {
  if (!state.imageElement || !detections) return;
  
  // Redraw the original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Define colors for different categories
  const colors = {
    'stud': '#00FF00', // Green
    'joist': '#FF0000', // Red
    'rafter': '#0000FF', // Blue
    'beam': '#00FFFF', // Cyan
    'plate': '#FF00FF', // Magenta
    'header': '#FFFF00', // Yellow
    'electrical_box': '#800080', // Purple
    'default': '#FF8C00' // Orange (default)
  };
  
  // Draw bounding boxes for each detection
  detections.forEach(det => {
    const [x, y, w, h] = det.bbox;
    const color = colors[det.category_name] || colors.default;
    
    // Draw rectangle
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    
    // Draw label
    ctx.fillStyle = color;
    const label = `${det.category_name} (${(det.confidence * 100).toFixed(0)}%)`;
    ctx.font = '14px Arial';
    
    // Draw background for text
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x, y - 20, textWidth + 10, 20);
    
    // Draw text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, x + 5, y - 5);
  });
}

function visualizeMeasurements(detections, measurements) {
  // First draw the detections
  visualizeDetections(detections);
  
  if (!measurements || !measurements.dimensions) return;
  
  // Add measurement visualizations
  const { dimensions, spacing } = measurements;
  
  // Draw dimensions for studs, joists, etc.
  for (const [category, data] of Object.entries(dimensions)) {
    if (!data.dimensions) continue;
    
    data.dimensions.forEach(dim => {
      const [x, y, w, h] = dim.bbox;
      
      // Draw dimension labels
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px Arial';
      
      // Format the dimensions
      const formattedSize = dim.standard_size !== 'custom' 
        ? dim.standard_size
        : `${dim.thickness.toFixed(1)}x${dim.depth.toFixed(1)}`;
      
      const label = `${formattedSize} ${measurements.unit}`;
      
      // Draw at the center of the bounding box
      const centerX = x + w/2;
      const centerY = y + h/2;
      
      // Add background for better visibility
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(centerX - textWidth/2 - 5, centerY - 8, textWidth + 10, 20);
      
      // Draw text
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.fillText(label, centerX, centerY + 5);
      ctx.textAlign = 'left'; // Reset alignment
    });
  }
  
  // Draw spacing measurements if available
  if (spacing && Object.keys(spacing.spacings || {}).length > 0) {
    for (const [category, data] of Object.entries(spacing.spacings)) {
      if (!data.center_points || data.center_points.length < 2) continue;
      
      // Get the center points and draw connections
      const points = data.center_points;
      const orientation = data.orientation;
      
      // Use different colors for spacing
      ctx.strokeStyle = category === 'stud' ? '#FFA500' : '#00CED1';
      ctx.setLineDash([5, 5]); // Dashed line for spacing
      
      for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i+1];
        
        // Draw line between centers
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        
        // Calculate midpoint
        const midX = (p1[0] + p2[0]) / 2;
        const midY = (p1[1] + p2[1]) / 2;
        
        // Get spacing value if available
        if (data.spacings && i < data.spacings.length) {
          const spacing = data.spacings[i].toFixed(1);
          const label = `${spacing} ${measurements.unit}`;
          
          // Background for spacing label
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
          
          if (orientation === 'vertical') {
            ctx.fillRect(midX - textWidth/2 - 5, midY - 8, textWidth + 10, 20);
          } else {
            ctx.fillRect(midX - textWidth/2 - 5, midY - 8, textWidth + 10, 20);
          }
          
          // Draw spacing label
          ctx.fillStyle = '#FFFFFF';
          ctx.textAlign = 'center';
          ctx.fillText(label, midX, midY + 5);
          ctx.textAlign = 'left'; // Reset alignment
        }
      }
      
      ctx.setLineDash([]); // Reset line style
    }
  }
}

// Calibration functions
function selectCalibrationMethod(event) {
  const method = event.target.dataset.method;
  
  // Hide all methods first
  document.querySelectorAll('.calibration-method').forEach(el => {
    el.style.display = 'none';
  });
  
  // Show selected method
  if (method === 'known-object') {
    document.getElementById('calibrateKnownObject').style.display = 'block';
  } else if (method === 'reference-points') {
    document.getElementById('calibrateReferencePoints').style.display = 'block';
    
    // Reset reference points
    state.referencePoints = [];
    document.getElementById('submitPointsBtn').disabled = true;
    
    // Ensure image is displayed
    if (state.imageElement) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(state.imageElement, 0, 0);
    }
  }
}

function handleObjectTypeChange() {
  const objectType = document.getElementById('objectType').value;
  const customDimensions = document.getElementById('customDimensions');
  
  if (objectType === 'custom') {
    customDimensions.style.display = 'block';
  } else {
    customDimensions.style.display = 'none';
  }
}

function startBoxDrawing() {
  // Reset box drawing state
  state.drawingBox = true;
  state.boxStart = null;
  
  // Redraw the original image
  if (state.imageElement) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.imageElement, 0, 0);
  }
  
  // Update button
  document.getElementById('drawBoxBtn').textContent = 'Drawing...';
}

function handleCanvasMouseDown(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  if (state.drawingBox) {
    // Start drawing box
    state.boxStart = [canvasX, canvasY];
  } else if (document.getElementById('calibrateReferencePoints').style.display === 'block') {
    // Add reference point (max 2)
    if (state.referencePoints.length < 2) {
      state.referencePoints.push([canvasX, canvasY]);
      
      // Draw the point
      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw line if we have 2 points
      if (state.referencePoints.length === 2) {
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(state.referencePoints[0][0], state.referencePoints[0][1]);
        ctx.lineTo(state.referencePoints[1][0], state.referencePoints[1][1]);
        ctx.stroke();
        
        // Enable submit button
        document.getElementById('submitPointsBtn').disabled = false;
      }
    }
  } else if (state.isDrawingPath) {
    // Add path point
    state.pathPoints.push([canvasX, canvasY]);
    
    // Redraw path
    drawPath();
    
    // Enable calculate button if we have at least 2 points
    document.getElementById('calculatePathBtn').disabled = state.pathPoints.length < 2;
  }
}

function handleCanvasMouseMove(event) {
  if (!state.drawingBox || !state.boxStart) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  // Redraw the image and the current box
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw the box
  const width = canvasX - state.boxStart[0];
  const height = canvasY - state.boxStart[1];
  
  ctx.strokeStyle = '#FF0000';
  ctx.lineWidth = 2;
  ctx.strokeRect(state.boxStart[0], state.boxStart[1], width, height);
}

function handleCanvasMouseUp(event) {
  if (!state.drawingBox || !state.boxStart) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  // Complete the box
  const width = canvasX - state.boxStart[0];
  const height = canvasY - state.boxStart[1];
  
  // Store the final box (ensure positive dimensions)
  const boxX = width >= 0 ? state.boxStart[0] : canvasX;
  const boxY = height >= 0 ? state.boxStart[1] : canvasY;
  const boxWidth = Math.abs(width);
  const boxHeight = Math.abs(height);
  
  // Store the box
  state.calibrationBox = [boxX, boxY, boxWidth, boxHeight];
  
  // Reset state
  state.drawingBox = false;
  state.boxStart = null;
  
  // Update button
  document.getElementById('drawBoxBtn').textContent = 'Draw Box';
  document.getElementById('submitKnownObjectBtn').disabled = false;
}

async function submitKnownObjectCalibration() {
  if (!state.calibrationBox || !state.currentImage) {
    showError('Please draw a box around a framing member');
    return;
  }
  
  try {
    // Get object dimensions
    const objectType = document.getElementById('objectType').value;
    let dimensions;
    
    if (objectType === 'custom') {
      const width = parseFloat(document.getElementById('customWidth').value);
      const height = parseFloat(document.getElementById('customHeight').value);
      dimensions = [width, height];
    } else {
      // Predefined dimensions
      switch (objectType) {
        case '2x4':
          dimensions = [1.5, 3.5];
          break;
        case '2x6':
          dimensions = [1.5, 5.5];
          break;
        case '2x8':
          dimensions = [1.5, 7.25];
          break;
        case '4x4':
          dimensions = [3.5, 3.5];
          break;
        default:
          dimensions = [1.5, 3.5]; // Default to 2x4
      }
    }
    
    // Submit calibration request
    const result = await window.api.calibrateImage(
      state.currentImage,
      'object',
      {
        bbox: state.calibrationBox,
        dimensions: dimensions,
        units: 'inches'
      }
    );
    
    // Store calibration data
    state.calibrationData = result;
    
    // Close modal and show success message
    hideModal(calibrateModal);
    alert('Calibration successful! The system is now calibrated for accurate measurements.');
    
    // Reprocess the image with calibration
    if (state.detectionResults) {
      await processImage();
    }
  } catch (error) {
    showError('Calibration failed: ' + error.message);
  }
}

async function submitPointsCalibration() {
  if (state.referencePoints.length !== 2 || !state.currentImage) {
    showError('Please place two reference points');
    return;
  }
  
  try {
    // Get distance between points
    const distance = parseFloat(document.getElementById('referenceDistance').value);
    const unit = document.getElementById('distanceUnit').value;
    
    // Submit calibration request
    const result = await window.api.calibrateImage(
      state.currentImage,
      'points',
      {
        points: state.referencePoints,
        distance: distance,
        units: unit
      }
    );
    
    // Store calibration data
    state.calibrationData = result;
    
    // Close modal and show success message
    hideModal(calibrateModal);
    alert('Calibration successful! The system is now calibrated for accurate measurements.');
    
    // Reprocess the image with calibration
    if (state.detectionResults) {
      await processImage();
    }
  } catch (error) {
    showError('Calibration failed: ' + error.message);
  }
}

// Path drawing functions
function startPathDrawing() {
  if (!state.currentImage) {
    showError('Please select an image first');
    return;
  }
  
  // Reset path points
  state.pathPoints = [];
  state.isDrawingPath = true;
  
  // Show path drawing modal
  showModal(pathModal);
  
  // Reset path results
  document.getElementById('pathResults').style.display = 'none';
  document.getElementById('calculatePathBtn').disabled = true;
  
  // Redraw canvas
  if (state.imageElement) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.imageElement, 0, 0);
  }
}

function drawPath() {
  if (state.pathPoints.length === 0 || !state.imageElement) return;
  
  // Redraw image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw path points and lines
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.fillStyle = '#FF0000';
  
  // Draw lines between points
  if (state.pathPoints.length > 1) {
    ctx.beginPath();
    ctx.moveTo(state.pathPoints[0][0], state.pathPoints[0][1]);
    
    for (let i = 1; i < state.pathPoints.length; i++) {
      ctx.lineTo(state.pathPoints[i][0], state.pathPoints[i][1]);
    }
    
    ctx.stroke();
  }
  
  // Draw points
  state.pathPoints.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Label points
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(index + 1, point[0], point[1] - 10);
    
    // Reset for next point
    ctx.fillStyle = '#FF0000';
  });
}

function clearPath() {
  state.pathPoints = [];
  drawPath();
  document.getElementById('calculatePathBtn').disabled = true;
}

function undoLastPathPoint() {
  if (state.pathPoints.length > 0) {
    state.pathPoints.pop();
    drawPath();
    document.getElementById('calculatePathBtn').disabled = state.pathPoints.length < 2;
  }
}

async function calculatePath() {
  if (state.pathPoints.length < 2 || !state.currentImage) {
    showError('Please place at least two path points');
    return;
  }
  
  try {
    // Show progress indicator
    document.getElementById('calculatePathBtn').textContent = 'Calculating...';
    document.getElementById('calculatePathBtn').disabled = true;
    
    // Submit path calculation request
    const result = await window.api.estimatePath(
      state.currentImage,
      state.pathPoints,
      state.calibrationData
    );
    
    // Display path results
    displayPathResults(result);
    
    // Reset button
    document.getElementById('calculatePathBtn').textContent = 'Calculate Path';
    document.getElementById('calculatePathBtn').disabled = false;
  } catch (error) {
    showError('Path calculation failed: ' + error.message);
    document.getElementById('calculatePathBtn').textContent = 'Calculate Path';
    document.getElementById('calculatePathBtn').disabled = false;
  }
}

function displayPathResults(pathResult) {
  if (!pathResult) return;
  
  const resultsDiv = document.getElementById('pathResults');
  const contentDiv = document.getElementById('pathResultsContent');
  
  // Build results HTML
  let html = `
    <div class="path-result-item">
      <strong>Total Distance:</strong> ${pathResult.display_distance} ${pathResult.display_unit}
    </div>
    <div class="path-result-item">
      <strong>Drill Points:</strong> ${pathResult.drill_count}
    </div>
  `;
  
  // Add segment details
  if (pathResult.path_segments && pathResult.path_segments.length > 0) {
    html += `<h4 style="margin-top: 1rem;">Path Segments:</h4>`;
    
    pathResult.path_segments.forEach((segment, index) => {
      html += `
        <div class="path-result-item">
          <span>Segment ${index + 1}:</span>
          <span>${segment.real_distance.toFixed(2)} ${segment.unit}</span>
        </div>
      `;
    });
  }
  
  // Display drill points if available
  if (pathResult.drill_points && pathResult.drill_points.length > 0) {
    html += `<h4 style="margin-top: 1rem;">Drilling Required:</h4>`;
    
    let totalDrillTime = 0;
    const drillTimes = {
      'easy': 3,
      'moderate': 5,
      'difficult': 8
    };
    
    pathResult.drill_points.forEach((point, index) => {
      const difficulty = point.difficulty || 'moderate';
      const drillTime = drillTimes[difficulty];
      totalDrillTime += drillTime;
      
      html += `
        <div class="path-result-item">
          <span>Drill ${index + 1}:</span>
          <span>${difficulty} (approx. ${drillTime} min)</span>
        </div>
      `;
    });
    
    html += `
      <div class="path-result-item" style="margin-top: 0.5rem;">
        <strong>Estimated Drilling Time:</strong> ${totalDrillTime} minutes
      </div>
    `;
  }
  
  // Display results
  contentDiv.innerHTML = html;
  resultsDiv.style.display = 'block';
  
  // Visualize path on canvas
  visualizePath(pathResult);
}

function visualizePath(pathResult) {
  if (!state.imageElement) return;
  
  // Redraw the original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw path
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 3;
  
  if (pathResult.path_segments && pathResult.path_segments.length > 0) {
    pathResult.path_segments.forEach(segment => {
      const start = segment.start;
      const end = segment.end;
      
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
      
      // Draw distance label
      const midX = (start[0] + end[0]) / 2;
      const midY = (start[1] + end[1]) / 2;
      
      const label = `${segment.real_distance.toFixed(1)} ${segment.unit}`;
      
      // Add background for visibility
      ctx.font = '14px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(midX - textWidth/2 - 5, midY - 10, textWidth + 10, 20);
      
      // Draw text
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.fillText(label, midX, midY + 5);
    });
  }
  
  // Draw drill points
  if (pathResult.drill_points && pathResult.drill_points.length > 0) {
    pathResult.drill_points.forEach(point => {
      const position = point.position;
      const difficulty = point.difficulty || 'moderate';
      
      // Colors based on difficulty
      let color;
      switch (difficulty) {
        case 'easy': color = '#00FF00'; break; // Green
        case 'moderate': color = '#FFA500'; break; // Orange
        case 'difficult': color = '#FF0000'; break; // Red
        default: color = '#FFA500'; // Default orange
      }
      
      // Draw drill point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(position[0], position[1], 7, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw outer ring
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(position[0], position[1], 12, 0, 2 * Math.PI);
      ctx.stroke();
    });
  }
  
  ctx.textAlign = 'left'; // Reset alignment
}

// Settings functions
function updateConfidenceLabel() {
  const value = document.getElementById('confidenceThreshold').value;
  document.getElementById('confidenceValue').textContent = value;
}

async function saveSettings() {
  const settings = {
    apiBaseUrl: document.getElementById('apiUrl').value,
    confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value),
    autoCalibrate: document.getElementById('autoCalibrate').checked,
    showMeasurements: document.getElementById('showMeasurements').checked
  };
  
  try {
    // Save settings through API
    const updatedConfig = await window.api.saveConfig(settings);
    
    // Update local state
    state.settings = updatedConfig.settings;
    
    // Close modal
    hideModal(settingsModal);
    
    // Show success
    alert('Settings saved successfully');
    
    // If we have results, redraw them with new settings
    if (state.detectionResults) {
      displayResults(state.detectionResults);
    }
  } catch (error) {
    showError('Failed to save settings: ' + error.message);
  }
}

// Report export
function exportReport() {
  if (!state.detectionResults) {
    showError('No results to export');
    return;
  }
  
  // Create report content
  const results = state.detectionResults;
  let report = 'Electrician Time Estimate Report\n';
  report += '==============================\n\n';
  
  report += `Estimated Time: ${results.time_estimate.estimated_minutes} minutes\n`;
  report += `Complexity: ${results.time_estimate.complexity}\n\n`;
  
  // Detection counts
  report += 'Detected Items:\n';
  const categories = {};
  results.detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  for (const [category, count] of Object.entries(categories)) {
    report += `${category}: ${count}\n`;
  }
  report += '\n';
  
  // Measurements
  if (results.measurements && results.measurements.dimensions) {
    report += 'Framing Measurements:\n';
    const { dimensions } = results.measurements;
    
    for (const [category, data] of Object.entries(dimensions)) {
      if (data.member_count > 0) {
        report += `${category}: ${data.most_common_size}\n`;
      }
    }
    report += '\n';
  }
  
  // Save the report
  try {
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'electrician_report.txt';
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  } catch (error) {
    showError('Failed to export report: ' + error.message);
  }
}

// Utility functions
function showModal(modal) {
  modal.style.display = 'block';
}

function hideModal(modal) {
  modal.style.display = 'none';
}

function showProgress(percent, message) {
  progressContainer.style.display = 'block';
  progressFill.style.width = `${percent}%`;
  progressText.textContent = message || 'Processing...';
}

function hideProgress() {
  progressContainer.style.display = 'none';
}

function showError(message) {
  alert('Error: ' + message);
  console.error(message);
}

// Initialize the app when loaded
document.addEventListener('DOMContentLoaded', init);


Now, let's add a package.json file for the Electron app:

json
{
  "name": "electrician-estimator",
  "version": "1.0.0",
  "description": "AI-assisted electrician time estimation application",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "dev": "electron . --dev",
    "build": "electron-builder"
  },
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "electron": "^25.3.1",
    "electron-builder": "^24.4.0"
  },
  "dependencies": {
    "axios": "^1.4.0",
    "form-data": "^4.0.0"
  },
  "build": {
    "appId": "com.electrician.estimator",
    "productName": "Electrician Estimator",
    "files": [
      "main.js",
      "preload.js",
      "src/**/*"
    ],
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": "nsis"
    },
    "mac": {
      "target": "dmg"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}


#Now, let's create a `run.py` file that will orchestrate everything:


# run.py
import os
import argparse
import subprocess
import sys
import signal
import threading
import time
import uvicorn
from pathlib import Path

from config.settings import API_HOST, API_PORT, API_DEBUG
from utils.logger import logger

def run_api():
    """
    Start the FastAPI server
    """
    logger.info(f"Starting Electrician Time Estimator API at {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG
    )

def run_frontend():
    """
    Start the Electron frontend
    """
    logger.info("Starting Electron frontend")
    frontend_path = Path(__file__).parent / "frontend" / "electron"
    
    # Check if we need to install dependencies
    if not (frontend_path / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
    
    # Run the Electron app
    process = subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def main():
    parser = argparse.ArgumentParser(description="Run the Electrician Time Estimator application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend")
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        if 'frontend_process' in locals():
            frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.api_only:
        # Run API only
        run_api()
    elif args.frontend_only:
        # Run frontend only
        frontend_process = run_frontend()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            frontend_process.terminate()
    else:
        # Run both
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Give the API a moment to start
        time.sleep(2)
        
        # Start the frontend
        frontend_process = run_frontend()
        
        try:
            # Monitor the frontend process
            while frontend_process.poll() is None:
                time.sleep(1)
                
            # If we get here, the frontend has ended
            logger.info("Frontend process ended")
            sys.exit(0)
        except KeyboardInterrupt:
            frontend_process.terminate()
            logger.info("Application terminated by user")

if __name__ == "__main__":
    main()


Let's also create a models directory file to ensure the model weights directory is created:


# models/__init__.py
"""
Machine learning models for the electrician time estimation application.
"""

from pathlib import Path

# Create model weights directory if it doesn't exist
weights_dir = Path(__file__).parent / "weights"
weights_dir.mkdir(exist_ok=True)

# Create empty __init__.py files in subdirectories if needed
detection_dir = Path(__file__).parent / "detection"
if detection_dir.exists() and not (detection_dir / "__init__.py").exists():
    with open(detection_dir / "__init__.py", "w") as f:
        f.write('"""Detection models for electrician time estimation."""\n')

# Create empty file in models/weights directory to ensure git tracks the directory
weights_gitkeep = weights_dir / ".gitkeep"
if not weights_gitkeep.exists():
    weights_gitkeep.touch()


Finally, let's create a small script to create necessary directories:


# create_dirs.py
"""
Create necessary directories for the electrician time estimation application.
"""

from pathlib import Path

def create_dirs():
    """Create the required directory structure."""
    
    # Base directories
    dirs = [
        "data/storage/images",
        "data/storage/processed",
        "data/storage/annotations",
        "data/storage/images/tasks",
        "models/weights",
        "models/weights/checkpoints",
        "models/weights/exports",
        "logs",
        "frontend/electron/src",
        "frontend/electron/src/styles",
        "frontend/electron/src/scripts",
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    # Ensure .gitkeep files exist where needed
    keeps = [
        "data/storage/images/.gitkeep",
        "data/storage/processed/.gitkeep",
        "data/storage/annotations/.gitkeep",
        "models/weights/.gitkeep",
    ]
    
    for keep_file in keeps:
        path = Path(keep_file)
        if not path.exists():
            path.touch()
            print(f"Created file: {path}")

if __name__ == "__main__":
    create_dirs()


Let's create the api/models directory with schemas for API requests:


# api/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

class CalibrationData(BaseModel):
    """Calibration data for measurement scaling."""
    pixels_per_unit: float
    unit: str
    confidence: float
    method: str
    reference_data: Dict[str, Any] = {}

class EstimationTask(BaseModel):
    """Task information for background processing."""
    id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    image_path: Optional[str] = None
    calibration_data: Optional[CalibrationData] = None
    results: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    """Response for a submitted processing task."""
    task_id: str
    status: str

class DetectionResult(BaseModel):
    """Object detection result."""
    id: int
    bbox: List[float]  # [x, y, width, height]
    category_id: int
    category_name: str
    confidence: float

class Point(BaseModel):
    """A 2D point."""
    x: float
    y: float

class PathSegment(BaseModel):
    """A segment of a wiring path."""
    start: List[float]  # [x, y]
    end: List[float]  # [x, y]
    pixel_distance: float
    real_distance: float
    unit: str

class DrillPoint(BaseModel):
    """A point where drilling is required."""
    position: List[float]  # [x, y]
    requires_drilling: bool
    category: Optional[str] = None
    difficulty: Optional[str] = None

class PathResult(BaseModel):
    """Result of a wiring path analysis."""
    status: str
    path_segments: List[PathSegment]
    total_distance: float
    display_distance: float
    display_unit: str
    unit: str
    drill_points: List[DrillPoint]
    drill_count: int


Let's integrate all these components together with a centralized configuration system for the application settings:


# config/settings.py (expanded version)
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application settings file
CONFIG_FILE = BASE_DIR / "app_config.json"

# Default settings
DEFAULT_CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False
    },
    "models": {
        "detection": {
            "model_type": "yolov8",
            "model_size": "m",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45
        },
        "measurements": {
            "auto_calibrate": True,
            "pixels_per_inch": None  # Will be set during calibration
        }
    },
    "paths": {
        "data_dir": str(BASE_DIR / "data" / "storage"),
        "model_weights_dir": str(BASE_DIR / "models" / "weights"),
        "logs_dir": str(BASE_DIR / "logs")
    },
    "complexity": {
        "thresholds": {
            "simple": 1.5,
            "moderate": 2.5
        },
        "base_time_minutes": {
            "simple": 7,
            "moderate": 12,
            "complex": 17
        }
    }
}

# Load config from file if it exists, otherwise use defaults
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        config = DEFAULT_CONFIG
else:
    config = DEFAULT_CONFIG
    # Save default config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError:
        pass

# Override with environment variables if provided
# API settings
API_HOST = os.getenv("API_HOST", config["api"]["host"])
API_PORT = int(os.getenv("API_PORT", config["api"]["port"]))
API_DEBUG = os.getenv("API_DEBUG", str(config["api"]["debug"])).lower() in ("true", "1", "t")

# Update config with env values
config["api"]["host"] = API_HOST
config["api"]["port"] = API_PORT
config["api"]["debug"] = API_DEBUG

# Machine learning settings
MODEL_TYPE = os.getenv("MODEL_TYPE", config["models"]["detection"]["model_type"])
MODEL_SIZE = os.getenv("MODEL_SIZE", config["models"]["detection"]["model_size"])
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 
                                      config["models"]["detection"]["confidence_threshold"]))
DEFAULT_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 
                                config["models"]["detection"]["iou_threshold"]))

# Update config with env values
config["models"]["detection"]["model_type"] = MODEL_TYPE
config["models"]["detection"]["model_size"] = MODEL_SIZE
config["models"]["detection"]["confidence_threshold"] = DEFAULT_CONFIDENCE_THRESHOLD
config["models"]["detection"]["iou_threshold"] = DEFAULT_IOU_THRESHOLD

# Path settings
DATA_DIR = Path(os.getenv("DATA_DIR", config["paths"]["data_dir"]))
MODEL_WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR", config["paths"]["model_weights_dir"]))
LOGS_DIR = Path(os.getenv("LOGS_DIR", config["paths"]["logs_dir"]))

# Update config with path settings
config["paths"]["data_dir"] = str(DATA_DIR)
config["paths"]["model_weights_dir"] = str(MODEL_WEIGHTS_DIR)
config["paths"]["logs_dir"] = str(LOGS_DIR)

# Derived directories
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
TASKS_DIR = IMAGES_DIR / "tasks"

# Job complexity settings from config
COMPLEXITY_THRESHOLDS = config["complexity"]["thresholds"]
BASE_TIME_MINUTES = config["complexity"]["base_time_minutes"]

# Ensure necessary directories exist
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Function to update and save config
def update_config(new_settings):
    """Update application configuration and save to file."""
    global config
    
    # Update config with new settings
    for section, values in new_settings.items():
        if section in config:
            if isinstance(values, dict):
                for key, value in values.items():
                    if key in config[section]:
                        config[section][key] = value
            else:
                config[section] = values
    
    # Save to file
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


Finally, let's create a simple install script that can be used to set up the application:


# install.py
"""
Installation script for the Electrician Time Estimator.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if the Python version is compatible."""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required.")
        return False
    return True

def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing Python dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install Node.js dependencies for the frontend."""
    frontend_path = Path("frontend") / "electron"
    
    if not frontend_path.exists():
        print(f"Frontend directory not found: {frontend_path}")
        return False
    
    print("Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing frontend dependencies: {e}")
        return False
    except FileNotFoundError:
        print("Node.js not found. Please install Node.js to run the frontend.")
        return False

def create_directory_structure():
    """Create the necessary directory structure."""
    print("Creating directory structure...")
    try:
        subprocess.run([sys.executable, "create_dirs.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating directories: {e}")
        return False

def download_model_weights():
    """Download pre-trained model weights (optional)."""
    print("Do you want to download pre-trained YOLOv8 weights? (y/n)")
    choice = input().lower()
    
    if choice in ('y', 'yes'):
        print("Downloading YOLOv8 weights...")
        try:
            # Download YOLOv8m.pt
            subprocess.run([
                sys.executable, 
                "-c", 
                "from ultralytics import YOLO; YOLO('yolov8m.pt')"
            ], check=True)
            print("Model weights downloaded successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model weights: {e}")
            print("You can still run the application, but you'll need to download weights later.")
            return False
    
    return True

def main():
    """Main installation function."""
    print("===== Electrician Time Estimator Installation =====")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Install frontend dependencies
    frontend_ok = install_frontend_dependencies()
    if not frontend_ok:
        print("Warning: Frontend dependencies couldn't be installed.")
        print("You can still run the API-only mode.")
    
    # Download model weights
    download_model_weights()
    
    print("\nInstallation completed!")
    print("\nTo run the application:")
    print("  - Full application: python run.py")
    print("  - API only: python run.py --api-only")
    if frontend_ok:
        print("  - Frontend only: python run.py --frontend-only")

if __name__ == "__main__":
    main()


With all these components, the end-to-end integration of the electrician time estimation application is complete. This integration connects the frontend UI to the backend API, ensures proper data flow between detection, estimation, and display components, implements error handling and recovery, adds progress indicators for long-running processes, and creates a configuration system for application settings.

The solution includes:

1. A robust backend API with FastAPI for handling image processing, framing detection, and time estimation
2. A user-friendly Electron frontend for interacting with the API
3. Background task processing for handling computation-intensive operations
4. Progress indicators to keep users informed during processing
5. Configuration management with both file and environment variable support
6. Error handling and recovery mechanisms
7. Installation and setup scripts for easy deployment

The application can be run in different modes (full, API-only, frontend-only) depending on the user's needs, and the code is structured to be maintainable and extensible for future improvements.
# End-to-End Integration for Electrician Time Estimator

I'll implement the code that integrates all components of the electrician time estimation application. This integration will connect the frontend UI to the backend API, ensure proper data flow between detection, estimation, and display components, implement error handling, add progress indicators, and create a configuration system.

## 1. API Routes Integration

First, let's create the API routes that will connect our detection and measurement modules:


# api/routes/estimation.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import tempfile
import json
import os
import shutil
from typing import Dict, List, Optional, Any

from models.detection import FramingDetector, detect_framing
from models.measurements import MeasurementEstimator
from utils.logger import get_logger
from utils.exceptions import EstimatorBaseException
from config.settings import (
    MODEL_WEIGHTS_DIR, IMAGES_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    COMPLEXITY_THRESHOLDS, BASE_TIME_MINUTES
)

logger = get_logger("api_estimation")
router = APIRouter()

# Task storage for background tasks
TASKS = {}

# Initialize models lazily
detector = None
estimator = None

def load_models():
    """
    Lazy-load models when needed.
    """
    global detector, estimator
    
    if detector is None:
        # Find best available model weights
        model_paths = list(MODEL_WEIGHTS_DIR.glob("framing_detector_*.pt"))
        if model_paths:
            model_path = sorted(model_paths)[-1]  # Use most recent model
            logger.info(f"Loading detection model from {model_path}")
            detector = FramingDetector.from_checkpoint(model_path)
        else:
            logger.warning("No detection model weights found, using pretrained model")
            detector = FramingDetector(pretrained=True)
    
    if estimator is None:
        estimator = MeasurementEstimator()
    
    return detector, estimator

def process_image_task(task_id: str, image_path: Path):
    """
    Background task to process an image.
    """
    try:
        # Update task status
        TASKS[task_id]["status"] = "processing"
        
        # Load the models
        detector, estimator = load_models()
        
        # Run detection
        detection_result = detect_framing(detector, str(image_path))
        TASKS[task_id]["progress"] = 40
        
        # If we have a stored calibration, use it
        calibration_data = TASKS[task_id].get("calibration_data")
        if calibration_data:
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        else:
            # Try to auto-calibrate using a known framing member
            if detection_result["detections"]:
                # Find a likely stud or joist for calibration
                for det in detection_result["detections"]:
                    if det["category_name"] in ["stud", "joist"] and det["confidence"] > 0.8:
                        # Use standard framing dimensions for calibration
                        try:
                            estimator.calibrate_from_known_object(
                                detection_result["image"],
                                det["bbox"],
                                (1.5, 3.5)  # 2x4 nominal dimensions
                            )
                            TASKS[task_id]["calibration_data"] = estimator.reference_scale.get_calibration_data()
                            break
                        except Exception as e:
                            logger.warning(f"Auto-calibration failed: {str(e)}")
        
        TASKS[task_id]["progress"] = 60
        
        # Run measurements
        measurement_result = estimator.analyze_framing_measurements(
            detection_result, calibration_check=False
        )
        TASKS[task_id]["progress"] = 80
        
        # Estimate time required based on complexity
        time_estimate = estimate_time(detection_result, measurement_result)
        
        # Save results
        results = {
            "detections": detection_result["detections"],
            "measurements": measurement_result["measurements"],
            "time_estimate": time_estimate
        }
        
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["results"] = results
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)

def estimate_time(detection_result: Dict, measurement_result: Dict) -> Dict:
    """
    Estimate time required for electrical work based on detection and measurements.
    """
    # Count different framing members
    member_counts = {}
    for det in detection_result["detections"]:
        cat = det["category_name"]
        member_counts[cat] = member_counts.get(cat, 0) + 1
    
    # Calculate complexity factors
    stud_count = member_counts.get("stud", 0)
    joist_count = member_counts.get("joist", 0)
    obstacle_count = member_counts.get("obstacle", 0) + member_counts.get("plumbing", 0)
    electrical_box_count = member_counts.get("electrical_box", 0)
    
    # Determine spacing complexity (irregular spacing is more complex)
    spacing_complexity = 0
    spacings = measurement_result.get("measurements", {}).get("spacing", {}).get("spacings", {})
    
    for cat, spacing_data in spacings.items():
        if not spacing_data.get("is_standard", True):
            spacing_complexity += 0.3
    
    # Calculate overall complexity score
    complexity_score = (
        stud_count * 0.05 +
        joist_count * 0.05 +
        obstacle_count * 0.2 +
        electrical_box_count * 0.1 +
        spacing_complexity
    )
    
    # Map to complexity level
    if complexity_score <= COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"

step 4 Enhance

Step 5 produce v1 (desktop application)
_______________________________________________________

We are on STEP 1

step 1 Guide for coding
create logic behind the idea, how ai factors in, math behind the complexities
_______________________________________________________


Examples of things to consider, but do not limit your ideas to.. Everything under this , is only an exmaple of the thought proccess, use your own thoughts to enhance upon that to begin step 1 of this goal


Framing
This indicates which part of the house the wire run is in (e.g.,
Basement, 1st Floor, 2nd Floor, Attic).



This specifies whether you're running through walls (studs),
ceilings/floors (joists), or roof (rafters/trusses).

Distance the wire is ran
This is the linear distance of the wire run within the framed section, measured in feet.

framing space
This is the spacing of the framing members in that location, measured in inches.

Complexeties
This describes various factors that may affect the complexity of the job, such as wood thickness/type, wire type/thickness, working space, obstructions, and drilling technique.

Follow these steps to estimate the labor time:

1. Calculate the number of framing members to drill through:
a. Convert the framing spacing from inches to feet.
b. Use the formula: Number of Members = CEILING(Wire Run Distance / Spacing in feet)
c. Round up to the nearest whole number.

2. Determine the complexity level based on the provided complexity factors:
- Simple: thin wire, easy drilling, clear space
- Moderate: standard wire, some obstructions, typical drilling
- Complex: thick wire, difficult drilling, many obstructions, tight space, thicker framing

3. Estimate the time per member based on the complexity level:
- Simple: 5-10 minutes per member
- Moderate: 10-15 minutes per member
- Complex: 15-20+ minutes per member

4. Calculate the total estimated time:
Total Estimated Time = Number of Members * Time per Member

Provide your answer in the following format:

Number of framing members to drill through: [number]
Complexity level: [Simple/Moderate/Complex]
Estimated time per member: [range in minutes]
Total estimated time: [range in minutes or hours]

Explanation: [Briefly explain your reasoning for the complexity level and time estimates, referencing the provided information]


# Step 1: Guide for Coding an AI-Assisted Electrician Time Estimation Application

## System Architecture Overview

The application will use computer vision and machine learning to analyze photos of residential framing members and estimate the labor time required for running electrical wiring. Here's the logical framework:

### Core Components:

1. **Photo Analysis Module**
- Processes uploaded images using computer vision
- Identifies framing members (studs, joists, rafters)
- Detects obstacles and existing utilities
- Measures distances and spacing

2. **Complexity Assessment Engine**
- Evaluates material types (wood species, engineered lumber, metal)
- Assesses space constraints and accessibility
- Identifies drilling challenges
- Calculates overall complexity score

3. **Time Estimation Algorithm**
- Calculates number of penetrations required
- Applies time factors based on complexity
- Generates time ranges with confidence levels

4. **User Interface**
- Photo upload capability
- Parameter input/confirmation
- Results display with detailed breakdown
- Saving/exporting functionality

## AI Integration

### Computer Vision Capabilities:
- **Object Detection**: Identify framing members, obstacles, and utilities
- **Semantic Segmentation**: Classify materials and spaces
- **Measurement Estimation**: Calculate distances and dimensions

### Model Selection:
- Primary model: YOLOv8 or Faster R-CNN for framing member detection
- Supporting models:
- Depth estimation network for 3D space understanding
- Material classification model for wood type identification
- Instance segmentation for obstacle detection

### Training Requirements:
- Labeled dataset of residential construction photos
- Augmentations for various lighting conditions
- Transfer learning from pretrained construction/building models

## Mathematical Framework

### 1. Framing Member Count Calculation

Number of Members = CEILING(Wire Run Distance / Spacing in feet)

Where:
- Wire Run Distance: Length of planned wire path (feet)
- Spacing in feet = Framing Spacing (inches) / 12

### 2. Complexity Scoring System
Start with base score of 1.0, then add factors:

**Material Factors:**
- Softwood (pine, spruce): +0.0
- Hardwood (oak, maple): +0.3
- Engineered lumber: +0.2
- Metal framing: +0.5

**Wire Factors:**
- Standard 14/12 AWG: +0.0
- 10 AWG or larger: +0.2
- Multiple wires: +0.2 per additional wire

**Workspace Factors:**
- Open accessible space: +0.0
- Confined space: +0.3
- Overhead work: +0.2
- Poor lighting: +0.2

**Obstruction Factors:**
- Clean framing: +0.0
- Existing utilities: +0.2 per type
- Insulation present: +0.2
- Structural elements to avoid: +0.3

**Final Complexity Level:**
- Simple: Score < 1.5
- Moderate: Score 1.5-2.5
- Complex: Score > 2.5

### 3. Time Estimation Formula

Per-Member Time = Base Time × Complexity Multiplier
Total Time = Number of Members × Per-Member Time


Where:
- Base Time:
- Simple: 7 minutes
- Moderate: 12 minutes
- Complex: 17 minutes
- Complexity Multiplier: Adjusts based on specific factors

### 4. Confidence Calculation

Confidence Score = (AI Detection Confidence × 0.6) + (Input Data Completeness × 0.4)

- Provide time ranges that widen as confidence decreases

## Technical Implementation Approach

### Backend Technology Stack:
- Python for AI/ML processing
- TensorFlow or PyTorch for computer vision models
- FastAPI for API development

### Frontend Options:
- Desktop: Electron for cross-platform compatibility
- Web application: React for responsive interface
- Mobile: React Native for on-site usage

### Development Phases:
1. Core AI model development and training
2. Calculation engine implementation
3. API layer for model interaction
4. User interface development
5. Testing and validation
6. Deployment and packaging

## Key Challenges and Solutions

1. **Variable Photo Quality**
- Implement image enhancement preprocessing
- Provide guidance for optimal photo angles
- Allow manual correction of AI detections

2. **Complex 3D Spaces**
- Request multiple photos from different angles
- Use depth estimation to understand spatial relationships
- Implement 3D reconstruction for complex layouts

3. **Regional Building Differences**
- Include region-specific framing databases
- Allow for custom parameter adjustments
- Train multiple regional models

4. **Confidence Handling**
- Clearly indicate low-confidence estimates
- Request additional information when needed
- Provide explanation of estimation factors

This framework provides the foundation for building an advanced AI-assisted application that can accurately estimate electrical wiring labor time based on construction photo analysis.




# AI-Assisted Electrician Time Estimation Application: Development Prompts

## Project Setup Prompts

### Prompt 1: Environment and Project Structure Setup

Create a Python project structure for an AI-assisted electrician time estimation application. The application should:
1. Use a virtual environment with Python 3.9+
2. Include directories for models, data, utils, api, and frontend
3. Set up a requirements.txt file with essential packages (TensorFlow/PyTorch, OpenCV, FastAPI, etc.)
4. Configure basic logging and error handling
5. Implement a modular architecture that separates concerns

Provide the complete directory structure and configuration files needed to begin development.


### Prompt 2: Data Collection and Preparation Framework

Develop a data handling module for the electrician time estimation application that can:
1. Process and store training images of residential framing members
2. Implement data augmentation for various lighting conditions and angles
3. Create annotation tools/scripts for labeling framing members, obstacles, and materials
4. Build a data pipeline that converts raw images to model-ready formats
5. Include functions to split data into training/validation/test sets

The module should handle common image formats and include proper error handling for corrupted files.


## AI Model Development Prompts

### Prompt 3: Framing Member Detection Model

Implement a computer vision model to detect and classify residential framing members from images. The model should:
1. Identify studs, joists, rafters, and other structural elements
2. Use a YOLOv8 or Faster R-CNN architecture with pretrained weights
3. Include training code with appropriate hyperparameters
4. Implement evaluation metrics (mAP, precision, recall)
5. Save model checkpoints and export to an inference-optimized format

Provide complete model definition, training loop, and inference functions with documentation.


### Prompt 4: Distance and Measurement Estimation

Create a module that estimates measurements from detected framing members in images. The module should:
1. Calculate the spacing between framing members
2. Estimate the dimensions of framing members (2x4, 2x6, etc.)
3. Implement a reference scale detection system or manual calibration
4. Calculate the total run distance for wiring paths
5. Provide confidence scores for measurements

Include visualization functions to display detected measurements on images for user verification.


### Prompt 5: Complexity Assessment System

Develop an algorithmic system that assesses the complexity of electrical wiring installation based on detected elements. The system should:
1. Analyze detected materials and assign appropriate difficulty scores
2. Identify and score obstacles and space constraints
3. Evaluate access difficulties from image context
4. Implement the complete complexity scoring formula from the specifications
5. Classify jobs into Simple, Moderate, and Complex categories

Provide the scoring system implementation with clear documentation of each factor's contribution.
\
## Backend Processing Prompts

### Prompt 6: Time Estimation Core Algorithm

Implement the core time estimation algorithm that calculates labor time based on detected elements and complexity scores. The algorithm should:
1. Calculate the number of framing members to drill through
2. Apply the appropriate time factors based on complexity
3. Generate time ranges with confidence intervals
4. Account for special conditions (overhead work, confined spaces)
5. Provide detailed breakdowns of time components

Include comprehensive unit tests to verify accuracy across various scenarios.


### Prompt 7: API Development

Create a RESTful API using FastAPI that serves the electrical time estimation functionality. The API should:
1. Accept image uploads and additional parameters
2. Process images through the detection and estimation pipeline
3. Return structured JSON responses with time estimates and confidence scores
4. Implement proper error handling and validation
5. Include authentication for production use
6. Provide API documentation using Swagger/OpenAPI

Ensure the API is properly structured with models, routes, and dependency injection.


## Frontend Development Prompts

### Prompt 8: Desktop Application UI (Electron)

Develop an Electron-based desktop application frontend for the electrician time estimation tool. The UI should:
1. Provide an intuitive image upload interface with drag-and-drop
2. Display detected framing members with visual overlays
3. Allow users to adjust detected elements if needed
4. Present time estimates with clear breakdowns and explanations
5. Enable saving/exporting of results in PDF and CSV formats
6. Implement responsive design for various screen sizes

Include all necessary HTML, CSS, and JavaScript files with proper documentation.


### Prompt 9: Parameter Input and Adjustment Interface

Create the user interface components for manual parameter input and adjustment. These should:
1. Allow users to specify job details not detectable from images
2. Provide forms for adjusting detected measurements if needed
3. Include material selection dropdowns with common options
4. Implement sliders for complexity factor adjustment
5. Show real-time updates to time estimates when parameters change

Ensure all inputs have appropriate validation and user feedback.


## Integration and Testing Prompts

### Prompt 10: End-to-End Integration

Implement code that integrates all components of the electrician time estimation application. This should:
1. Connect the frontend UI to the backend API
2. Ensure proper data flow between detection, estimation, and display components
3. Implement error handling and recovery throughout the pipeline
4. Add progress indicators for long-running processes
5. Create a configuration system for application settings

Include comprehensive integration tests that verify the complete workflow.


### Prompt 11: Testing Framework and Validation

Develop a comprehensive testing framework for the application that includes:
1. Unit tests for individual components and algorithms
2. Integration tests for component interactions
3. End-to-end tests simulating real user workflows
4. Performance benchmarking for processing times
5. Accuracy validation against known examples with ground truth
6. Test fixtures and mock data generation utilities

Ensure test coverage across all critical components with automated reporting.


## Deployment and Enhancement Prompts

### Prompt 12: Application Packaging and Deployment

Create the necessary scripts and configuration to package the application for distribution. This should:
1. Bundle the Electron application with all dependencies
2. Configure proper versioning and update mechanisms
3. Implement installer generation for Windows/Mac/Linux
4. Set up proper logging and error reporting for production use
5. Include documentation and help resources within the application

Provide complete deployment instructions and verification procedures.


### Prompt 13: Performance Optimization

Implement optimizations to enhance the performance and user experience of the application:
1. Model quantization techniques to speed up inference
2. Image preprocessing optimizations for faster detection
3. Caching mechanisms for repeated calculations
4. Background processing for time-intensive tasks
5. UI optimizations for responsiveness
6. Memory management improvements

Include before/after benchmarks demonstrating the performance gains.


### Prompt 14: Enhancement for Mobile Use

Extend the application to support mobile usage through:
1. A React Native implementation for iOS and Android
2. Camera integration for direct photo capture
3. Cloud synchronization of estimates and projects
4. Offline functionality for job sites without connectivity
5. Touch-optimized UI adjustments
6. Battery usage optimizations

Provide the additional code and modifications needed while maintaining the core estimation logic.


## Advanced Features Prompts

### Prompt 15: 3D Space Reconstruction

Implement advanced 3D reconstruction capabilities that:
1. Process multiple images to create a 3D understanding of the space
2. Generate a basic 3D model of the framing structure
3. Allow users to trace wiring paths through the 3D model
4. Calculate optimal routing to minimize drilling
5. Visualize the complete wiring path in 3D

Include algorithms for image registration, depth estimation, and 3D visualization.


### Prompt 16: Report Generation and Documentation

Create a comprehensive reporting module that:
1. Generates detailed PDF reports of time estimations
2. Includes annotated images highlighting detection results
3. Provides material lists and drilling specifications
4. Creates project documentation for clients and crew
5. Supports customizable templates and branding
6. Implements export to various formats (PDF, Excel, CSV)

Provide the complete reporting implementation with example outputs.


These structured prompts break down the development process into manageable sections, ensuring that each component is properly implemented before integration. Following this sequence will lead to a stable, well-architected application that meets the requirements.
"""""
""""
Prompt 1

# Electrician Time Estimator: Project Structure Setup

Below is a complete project structure and configuration for the AI-assisted electrician time estimation application.

## Directory Structure


electrician-estimator/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── run.py
├── config/
│ ├── __init__.py
│ ├── logging_config.py
│ └── settings.py
├── api/
│ ├── __init__.py
│ ├── main.py
│ ├── routes/
│ │ ├── __init__.py
│ │ └── estimation.py
│ ├── models/
│ │ ├── __init__.py
│ │ └── schemas.py
│ └── middleware/
│ ├── __init__.py
│ └── error_handling.py
├── data/
│ ├── __init__.py
│ ├── loaders/
│ │ ├── __init__.py
│ │ └── image_loader.py
│ ├── processors/
│ │ ├── __init__.py
│ │ └── image_processor.py
│ └── storage/
│ ├── images/
│ ├── processed/
│ └── annotations/
├── models/
│ ├── __init__.py
│ ├── detection/
│ │ ├── __init__.py
│ │ └── framing_detector.py
│ ├── estimation/
│ │ ├── __init__.py
│ │ └── time_estimator.py
│ └── weights/
├── utils/
│ ├── __init__.py
│ ├── exceptions.py
│ ├── logger.py
│ └── visualization.py
├── frontend/
│ └── electron/
│ ├── package.json
│ ├── main.js
│ └── src/
│ ├── index.html
│ ├── styles/
│ │ └── main.css
│ └── scripts/
│ └── main.js
└── tests/
├── __init__.py
├── conftest.py
├── test_api/
│ ├── __init__.py
│ └── test_estimation.py
├── test_models/
│ ├── __init__.py
│ └── test_detection.py
└── test_utils/
├── __init__.py
└── test_visualization.py
"""
</sourcefile>"""
## Configuration Files

### requirements.txt

# Core dependencies
python-dotenv>=0.19.0
pydantic>=1.9.0

# API
fastapi>=0.88.0
uvicorn>=0.20.0
python-multipart>=0.0.5
httpx>=0.23.0

# Machine Learning & Computer Vision
tensorflow>=2.11.0 # For CPU
# tensorflow-gpu>=2.11.0 # Uncomment for GPU support
# OR
# torch>=1.13.0
# torchvision>=0.14.0
opencv-python>=4.7.0
pillow>=9.4.0
numpy>=1.23.5
scikit-image>=0.19.3
scikit-learn>=1.2.0
matplotlib>=3.6.3

# Image processing
albumentations>=1.3.0

# Utilities
tqdm>=4.64.1
loguru>=0.6.0

# Testing
pytest>=7.2.1
pytest-cov>=4.0.0

# Development
black>=23.1.0
flake8>=6.0.0
isort>=5.12.0


### .env.example

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=True

# Model Settings
MODEL_TYPE=tensorflow # or pytorch
DETECTION_MODEL=yolov8
CONFIDENCE_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO


### .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Environment variables
.env

# Model weights
models/weights/*.h5
models/weights/*.pth
models/weights/*.pb
models/weights/*.onnx

# Data files
data/storage/images/*
data/storage/processed/*
data/storage/annotations/*
data/storage/images/.gitkeep
data/storage/processed/.gitkeep
data/storage/annotations/.gitkeep

# Log files
logs/
*.log

# Test coverage
.coverage
htmlcov/

# Node.js dependencies
node_modules/

## Core Configuration Files

### config/logging_config.py

import logging
import os
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

def configure_logging(name="electrician_estimator", level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        name (str): Logger name
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers if they don't exist yet
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(logs_dir / f"{name}.log")
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter and add it to the handlers
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
    
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
    
        return logger
        
        # Load image if path is provided
        def load_and_transform_image(image, custom_transform, save_path=None):
            if isinstance(image, (str, Path)):
                img = load_image(image)
            else:
                img = image.copy()
        
            # Apply transformation
            augmented = custom_transform(image=img)['image']
        
            # Save if path is provided
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(augmented, save_path)
        
            return augmented
        
        def get_image_dimensions(first_img):
            height, width = first_img.shape[:2]

                # Determine shape based on normalization
                if normalize:
                    batch = np.zeros((len(image_paths), height, width, 3), dtype=np.float32)
                else:
                    batch = np.zeros((len(image_paths), height, width, 3), dtype=np.uint8)

                # Process each image
                for i, img_path in enumerate(image_paths):
                    try:
                        processed_img = self.image_processor.preprocess_image(
                            img_path,
                            normalize=normalize,
                            enhance_contrast=True,
                            denoise=True
                        )

                        batch[i] = processed_img

                    except Exception as e:
                        logger.error(f"Failed to prepare image {img_path}: {str(e)}")
                        # Fill with zeros for failed images
                        batch[i] = np.zeros((height, width, 3), dtype=batch.dtype)

                return batch

# Load model
try:
    if weights_path is not None:
        # Load custom trained weights
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ModelNotFoundError(f"Weights file not found: {weights_path}")

        logger.info(f"Loading custom weights from {weights_path}")
        self.model = YOLO(str(weights_path))

    elif pretrained:
        # Load pretrained weights
        logger.info(f"Loading pretrained YOLOv8{model_size}")
        self.model = YOLO(f"yolov8{model_size}.pt")

        # Update number of classes if needed
        if self.model.names != YOLO_CONFIG['class_names']:
            logger.info(f"Updating model for {len(CATEGORIES)} framing categories")
            self.model.names = YOLO_CONFIG['class_names']
    else:
        # Initialize with random weights
        logger.info(f"Initializing YOLOv8{model_size} with random weights")
        self.model = YOLO(f"yolov8{model_size}.yaml")

    # Move model to device
    self.model.to(self.device)

except Exception as e:
    raise ModelNotFoundError(f"Failed to load YOLOv8 model: {str(e)}")

def detect(
    self,
    image: Union[str, Path, np.ndarray],
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    return_original: bool = False
) -> Dict:
    """
    Detect framing members in an image.

    Args:
        image: Image file path or numpy array
        conf_threshold: Confidence threshold for detections (overrides default)
        iou_threshold: IoU threshold for NMS (overrides default)
        return_original: Whether to include the original image in the results

    Returns:
        Dict: Detection results with keys:
        - 'detections': List of detection dictionaries
        - 'image': Original image (if return_original=True)
        - 'inference_time': Time taken for inference
    """
    # Use specified thresholds or fall back to instance defaults
    conf = conf_threshold if conf_threshold is not None else self.conf_threshold
    iou = iou_threshold if iou_threshold is not None else self.iou_threshold

    try:
        # Track inference time
        start_time = time.time()

        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        inference_time = time.time() - start_time

        # Process results
        detections = []

        # Extract results from the first image (or only image)
        result = results[0]

        # Convert boxes to the desired format
        if len(result.boxes) > 0:
            # Get boxes, classes, and confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            # Format detections
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                x1, y1, x2, y2 = box

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # Get class name
                class_name = result.names[cls]

                detection = {
                    'id': i,
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'category_id': int(cls),
                    'category_name': class_name,
                    'confidence': float(score)
                }

                detections.append(detection)

        # Prepare return dictionary
        results_dict = {
            'detections': detections,
            'inference_time': inference_time
        }

        # Include original image if requested
        if return_original:
            if isinstance(image, (str, Path)):
                # If image is a path, get the processed image from results
                results_dict['image'] = result.orig_img
            else:
                # If image is an array, use it directly
                results_dict['image'] = image

        return results_dict

    except Exception as e:
        raise ModelInferenceError(f"Error during framing detection: {str(e)}")

def train(
    self,
    data_yaml: Union[str, Path],
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    project: str = 'framing_detection',
    name: str = 'train',
    device: Optional[str] = None,
    lr0: float = 0.01,
    lrf: float = 0.01,
    save: bool = True,
    resume: bool = False,
    pretrained: bool = True,
    **kwargs
) -> Any:
    """
    Train the detector on a dataset.

    Args:
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        patience: Epochs to wait for no improvement before early stopping
        project: Project name for saving results
        name: Run name for this training session
        device: Device to use (None for auto-detection)
        lr0: Initial learning rate
        lrf: Final learning rate (fraction of lr0)
        save: Whether to save the model
        resume: Resume training from the last checkpoint
        pretrained: Use pretrained weights
        **kwargs: Additional arguments to pass to the trainer

    Returns:
        Training results
    """
    device = device or self.device

    logger.info(f"Training YOLOv8{self.model_size} on {device}")
    logger.info(f"Data config: {data_yaml}, Epochs: {epochs}, Batch size: {batch_size}")

    # Set up training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,
        'project': project,
        'name': name,
        'device': device,
        'lr0': lr0,
        'lrf': lrf,
        'save': save,
        'pretrained': pretrained,
        'resume': resume
    }

    # Add any additional kwargs
    train_args.update(kwargs)

    # Start training
    try:
        results = self.model.train(**train_args)

        logger.info(f"Training completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def export(
    self,
    format: str = 'onnx',
    output_path: Optional[Union[str, Path]] = None,
    dynamic: bool = True,
    half: bool = True,
    simplify: bool = True
) -> Path:
    """
    Export the model to a deployable format.

    Args:
        format: Export format ('onnx', 'torchscript', 'openvino', etc.)
        output_path: Path to save the exported model
        dynamic: Use dynamic axes in ONNX export
        half: Export with half precision (FP16)
        simplify: Simplify the model during export

    Returns:
        Path: Path to the exported model
    """
    if output_path is None:
        # Generate default output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"framing_detector_{timestamp}.{format}"
    else:
        output_path = Path(output_path)

    logger.info(f"Exporting model to {format} format: {output_path}")

    try:
        # Export the model
        exported_path = self.model.export(
            format=format,
            imgsz=self.img_size,
            dynamic=dynamic,
            half=half,
            simplify=simplify
        )

        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If the export path is different from the desired output path, move it
        if str(exported_path) != str(output_path):
            shutil.copy(exported_path, output_path)
            os.remove(exported_path)
            logger.info(f"Moved exported model to {output_path}")

        logger.info(f"Model exported successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise

def save_checkpoint(
    self,
    path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> Path:
    """
    Save a model checkpoint.

    Args:
        path: Path to save the checkpoint
        overwrite: Whether to overwrite if file exists

    Returns:
        Path: Path to the saved checkpoint
    """
    if path is None:
        # Generate default path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = CHECKPOINTS_DIR / f"framing_detector_{timestamp}.pt"
    else:
        path = Path(path)

    # Check if file exists and overwrite is False
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint file already exists: {path}")

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model checkpoint to {path}")

    try:
        self.model.save(str(path))
        logger.info(f"Model checkpoint saved successfully")
        return path
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise

@classmethod
def from_checkpoint(
    cls,
    checkpoint_path: Union[str, Path],
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> 'FramingDetector':
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        FramingDetector: Loaded model
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise ModelNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Create detector with custom weights
    detector = cls(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        pretrained=False,
        weights_path=checkpoint_path
    )

    return detector

print(f"mAP: {metrics['mAP']:.4f}")

"""""

This implementation provides a complete and flexible framing member detection system based on YOLOv8. The model can identify various framing elements in residential construction images, which is essential for the electrician time estimation application. The code includes comprehensive training, evaluation, and inference capabilities, along with utilities for saving, loading, and exporting models.





Prompt 4


# Measurement Estimation Module for Framing Member Detection

This module provides comprehensive functionality for estimating real-world measurements from detected framing members in residential construction images.

## Directory Structure


models/
└── measurements/
├── __init__.py
├── measurement_estimator.py
├── reference_scale.py
├── spacing_calculator.py
├── dimension_estimator.py
├── path_calculator.py
└── visualization.py

"""
## Implementation Files

### models/measurements/\_\_init\_\_.py

""""
Measurement estimation module for the electrician time estimation application.
This module provides tools for calculating distances, dimensions, and wiring paths
from detected framing members in residential construction images.
"""

from models.measurements.measurement_estimator import MeasurementEstimator
from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from models.measurements.visualization import (
visualize_measurements,
visualize_wiring_path,
visualize_scale_calibration
)

__all__ = [
'MeasurementEstimator',
'ReferenceScale',
'ScaleCalibration',
'SpacingCalculator',
'DimensionEstimator',
'PathCalculator',
'visualize_measurements',
'visualize_wiring_path',
'visualize_scale_calibration'
]


### models/measurements/measurement_estimator.py

"""
Main measurement estimation class for analyzing framing members.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json

from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("measurement_estimator")

class MeasurementEstimator:
    """
    Class for estimating measurements from detected framing members.
    """

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None,
        confidence_threshold: float = 0.7,
        detection_threshold: float = 0.25
    ):
        """
        Initialize the measurement estimator.

        Args:
            pixels_per_inch: Calibration value (pixels per inch)
            calibration_data: Pre-computed calibration data
            confidence_threshold: Threshold for including detections in measurements
            detection_threshold: Threshold for detection confidence
        """
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold

        # Initialize the reference scale
        self.reference_scale = ReferenceScale(
            pixels_per_inch=pixels_per_inch,
            calibration_data=calibration_data
        )

        # Initialize measurement components
        self.spacing_calculator = SpacingCalculator(self.reference_scale)
        self.dimension_estimator = DimensionEstimator(self.reference_scale)
        self.path_calculator = PathCalculator(self.reference_scale)

        # Store measurement history
        self.last_measurement_result = None

        logger.info("Measurement estimator initialized")

    def calibrate_from_reference(
        self,
        image: np.ndarray,
        reference_points: List[Tuple[int, int]],
        reference_distance: float,
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using reference points.

        Args:
            image: Input image array
            reference_points: List of two (x, y) points defining a known distance
            reference_distance: Known distance between points
            units: Units of the reference distance ("inches", "feet", "mm", "cm", "m")

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_points(
                reference_points, reference_distance, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def calibrate_from_known_object(
        self,
        image: np.ndarray,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using a known object.

        Args:
            image: Input image array
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            units: Units of the reference dimensions

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_object(
                object_bbox, object_dimensions, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def analyze_framing_measurements(
        self,
        detection_result: Dict,
        calibration_check: bool = True
    ) -> Dict:
        """
        Analyze framing member detections to extract measurements.

        Args:
            detection_result: Detection results from framing detector
            calibration_check: Whether to verify calibration first

        Returns:
            Dict: Measurement analysis results
        """
        if calibration_check and not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Filter detections based on confidence
            detections = [det for det in detection_result['detections']
                          if det['confidence'] >= self.detection_threshold]

            if not detections:
                logger.warning("No valid detections found for measurement analysis")
                return {
                    "status": "warning",
                    "message": "No valid detections found",
                    "measurements": {}
                }

            # Extract image if available
            image = detection_result.get('image')

            # Calculate spacing measurements
            spacing_results = self.spacing_calculator.calculate_spacings(detections, image)

            # Estimate framing dimensions
            dimension_results = self.dimension_estimator.estimate_dimensions(detections, image)

            # Collect all measurements and calculate overall confidence
            measurements = {
                "spacing": spacing_results,
                "dimensions": dimension_results,
                "unit": self.reference_scale.get_unit(),
                "pixels_per_unit": self.reference_scale.get_pixels_per_unit()
            }

            # Calculate overall confidence score
            detection_confs = [det['confidence'] for det in detections]
            avg_detection_conf = sum(detection_confs) / len(detection_confs) if detection_confs else 0

            spacing_conf = spacing_results.get("confidence", 0) if spacing_results else 0
            dimension_conf = dimension_results.get("confidence", 0) if dimension_results else 0
            scale_conf = self.reference_scale.get_calibration_confidence()

            overall_confidence = 0.4 * avg_detection_conf + 0.3 * spacing_conf + \
                               0.2 * dimension_conf + 0.1 * scale_conf

            # Store the results for later reference
            self.last_measurement_result = {
                "status": "success",
                "message": "Measurement analysis completed",
                "measurements": measurements,
                "confidence": overall_confidence
            }

            return self.last_measurement_result

        except Exception as e:
            error_msg = f"Measurement analysis failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def estimate_wiring_path(
        self,
        detection_result: Dict,
        path_points: List[Tuple[int, int]],
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Estimate the total distance of a wiring path.

        Args:
            detection_result: Detection results from framing detector
            path_points: List of (x, y) points defining the wiring path
            drill_points: List of (x, y) points where drilling is required

        Returns:
            Dict: Wiring path analysis
        """
        if not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Extract image if available
            image = detection_result.get('image')

            # Calculate path measurements
            path_results = self.path_calculator.calculate_path(
                path_points, image, drill_points=drill_points
            )

            # Calculate drill points if not provided
            if drill_points is None and 'detections' in detection_result:
                detected_drill_points = self.path_calculator.identify_drill_points(
                    path_points, detection_result['detections'], image
                )
                path_results['detected_drill_points'] = detected_drill_points

            # Store the results
            self.last_path_result = path_results

            return path_results

        except Exception as e:
            error_msg = f"Wiring path estimation failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def save_measurements(self, output_file: Union[str, Path]) -> None:
        """
        Save the last measurement results to a file.

        Args:
            output_file: Path to save the measurement data
        """
        if self.last_measurement_result is None:
            logger.warning("No measurements to save")
            return

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w') as f:
                json.dump(self.last_measurement_result, f, indent=2)

            logger.info(f"Measurement results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save measurements: {str(e)}")

    def load_measurements(self, input_file: Union[str, Path]) -> Dict:
        """
        Load measurement results from a file.

        Args:
            input_file: Path to the measurement data file

        Returns:
            Dict: Loaded measurement data
        """
        input_file = Path(input_file)

        if not input_file.exists():
            error_msg = f"Measurement file not found: {input_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(input_file, 'r') as f:
                measurement_data = json.load(f)

            self.last_measurement_result = measurement_data

            # Update calibration if present
            if 'measurements' in measurement_data and 'pixels_per_unit' in measurement_data['measurements']:
                unit = measurement_data['measurements'].get('unit', 'inches')
                pixels_per_unit = measurement_data['measurements']['pixels_per_unit']

                self.reference_scale.set_calibration(pixels_per_unit, unit)

            logger.info(f"Measurement results loaded from {input_file}")
            return measurement_data

        except Exception as e:
            def identify_drill_points(
                self,
                path_points: List[Tuple[float, float]],
                detections: List[Dict],
                image: Optional[np.ndarray] = None
            ) -> List[Dict]:
                """
                Identify points where drilling is required along a wiring path.

                Args:
                    path_points: List of (x, y) points defining the wiring path
                    detections: List of detection dictionaries
                    image: Original image (optional, for visualization)

                Returns:
                    List[Dict]: List of drill points
                """
                if len(path_points) < 2:
                    return []

                drill_points = []

                # Process each path segment
                for i in range(len(path_points) - 1):
                    start_x, start_y = path_points[i]
                    end_x, end_y = path_points[i+1]

                    # Check intersection with each framing member
                    for det in detections:
                        # Skip non-framing categories
                        category = det['category_name']
                        if category not in ['stud', 'joist', 'rafter', 'beam', 'plate', 'header']:
                            continue

                        # Get bounding box
                        bbox = det['bbox']
                        x, y, w, h = bbox

                        # Check if segment intersects the bounding box
                        if self._segment_intersects_box(start_x, start_y, end_x, end_y, x, y, w, h):
                            # Calculate intersection point
                            intersection = self._get_segment_box_intersection(
                                start_x, start_y, end_x, end_y, x, y, w, h
                            )

                            if intersection:
                                # Determine drill difficulty based on member type and size
                                difficulty = self._calculate_drill_difficulty(det)

                                drill_points.append({
                                    "position": intersection,
                                    "requires_drilling": True,
                                    "category": category,
                                    "difficulty": difficulty
                                })

                return drill_points

            def _segment_intersects_box(
                self,
                start_x: float,
                start_y: float,
                end_x: float,
                end_y: float,
                box_x: float,
                box_y: float,
                box_width: float,
                box_height: float
            ) -> bool:
                """
                Check if a line segment intersects with a bounding box.

                Args:
                    start_x, start_y: Start point of segment
                    end_x, end_y: End point of segment
                    box_x, box_y, box_width, box_height: Bounding box

                Returns:
                    bool: True if the segment intersects the box
                """
                # Define box corners
                left = box_x
                right = box_x + box_width
                top = box_y
                bottom = box_y + box_height

                # Check if either endpoint is inside the box
                if (left <= start_x <= right and top <= start_y <= bottom) or \
                   (left <= end_x <= right and top <= end_y <= bottom):
                    return True

                # Check if line segment intersects any of the box edges
                edges = [
                    (left, top, right, top),          # Top edge
                    (right, top, right, bottom),      # Right edge
                    (left, bottom, right, bottom),    # Bottom edge
                    (left, top, left, bottom)         # Left edge
                ]

                for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
                    if self._line_segments_intersect(
                        start_x, start_y, end_x, end_y,
                        edge_x1, edge_y1, edge_x2, edge_y2
                    ):
                        return True

                return False

            def _line_segments_intersect(
                self,
                a_x1: float, a_y1: float, a_x2: float, a_y2: float,
                b_x1: float, b_y1: float, b_x2: float, b_y2: float
            ) -> bool:
                """
                Check if two line segments intersect.

                Args:
                    a_x1, a_y1, a_x2, a_y2: First line segment
                    b_x1, b_y1, b_x2, b_y2: Second line segment

                Returns:
                    bool: True if the segments intersect
                """
                # Calculate the direction vectors
                r = (a_x2 - a_x1, a_y2 - a_y1)
                s = (b_x2 - b_x1, b_y2 - b_y1)

                # Calculate the cross product (r × s)
                rxs = r[0] * s[1] - r[1] * s[0]

                # If r × s = 0, the lines are collinear or parallel
                if abs(rxs) < 1e-8:
                    return False

                # Calculate t and u parameters
                q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
                t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
                u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

                # Check if intersection point is within both segments
                return 0 <= t <= 1 and 0 <= u <= 1

            def _get_segment_box_intersection(
                self,
                start_x: float,
                start_y: float,
                end_x: float,
                end_y: float,
                box_x: float,
                box_y: float,
                box_width: float,
                box_height: float
            ) -> Optional[Tuple[float, float]]:
                """
                Get the intersection point of a line segment with a box.

                Args:
                    start_x, start_y: Start point of segment
                    end_x, end_y: End point of segment
                    box_x, box_y, box_width, box_height: Bounding box

                Returns:
                    Optional[Tuple[float, float]]: Intersection point or None
                """
                # Define box corners
                left = box_x
                right = box_x + box_width
                top = box_y
                bottom = box_y + box_height

                # If start point is inside the box, use it
                if left <= start_x <= right and top <= start_y <= bottom:
                    return (start_x, start_y)

                # If end point is inside the box, use it
                if left <= end_x <= right and top <= end_y <= bottom:
                    return (end_x, end_y)

                # Check intersections with box edges
                edges = [
                    (left, top, right, top),          # Top edge
                    (right, top, right, bottom),      # Right edge
                    (left, bottom, right, bottom),    # Bottom edge
                    (left, top, left, bottom)         # Left edge
                ]

                for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
                    if self._line_segments_intersect(
                        start_x, start_y, end_x, end_y,
                        edge_x1, edge_y1, edge_x2, edge_y2
                    ):
                        # Calculate intersection point
                        intersection = self._calculate_intersection_point(
                            start_x, start_y, end_x, end_y,
                            edge_x1, edge_y1, edge_x2, edge_y2
                        )

                        if intersection:
                            return intersection

                return None

            def _calculate_intersection_point(
                self,
                a_x1: float, a_y1: float, a_x2: float, a_y2: float,
                b_x1: float, b_y1: float, b_x2: float, b_y2: float
            ) -> Optional[Tuple[float, float]]:
                """
                Calculate the intersection point of two line segments.

                Args:
                    a_x1, a_y1, a_x2, a_y2: First line segment
                    b_x1, b_y1, b_x2, b_y2: Second line segment

                Returns:
                    Optional[Tuple[float, float]]: Intersection point or None
                """
                # Calculate the direction vectors
                r = (a_x2 - a_x1, a_y2 - a_y1)
                s = (b_x2 - b_x1, b_y2 - b_y1)

                # Calculate the cross product (r × s)
                rxs = r[0] * s[1] - r[1] * s[0]

                # If r × s = 0, the lines are collinear or parallel
                if abs(rxs) < 1e-8:
                    return None

                # Calculate t parameter
                q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
                t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
                u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

                # Check if intersection point is within both segments
                if 0 <= t <= 1 and 0 <= u <= 1:
                    # Calculate intersection point
                    ix = a_x1 + t * r[0]
                    iy = a_y1 + t * r[1]
                    return (ix, iy)

                return None

            def _calculate_drill_difficulty(self, detection: Dict) -> str:
                """
                Estimate the difficulty of drilling through a detected member.

                Args:
                    detection: Detection dictionary

                Returns:
                    str: Difficulty level ("easy", "moderate", "difficult")
                """
                category = detection['category_name']

                # Estimate based on category
                if category in ['stud', 'plate']:
                    base_difficulty = "easy"
                elif category in ['joist', 'rafter']:
                    base_difficulty = "moderate"
                elif category in ['beam', 'header']:
                    base_difficulty = "difficult"
                else:
                    base_difficulty = "moderate"

                # Adjust based on member size if available
                if 'dimensions' in detection:
                    thickness = detection['dimensions'].get('thickness', 0)

                    # Convert to inches for standard comparison
                    if self.reference_scale.get_unit() != "inches":
                        thickness_inches = self.reference_scale.convert_units(
                            thickness, self.reference_scale.get_unit(), "inches"
                        )
                    else:
                        thickness_inches = thickness

                    # Adjust difficulty based on thickness
                    if thickness_inches > 3.0:  # Thick member
                        if base_difficulty == "easy":
                            base_difficulty = "moderate"
                        elif base_difficulty == "moderate":
                            base_difficulty = "difficult"
                    elif thickness_inches < 1.0:  # Thin member
                        if base_difficulty == "difficult":
                            base_difficulty = "moderate"
                        elif base_difficulty == "moderate":
                            base_difficulty = "easy"

                return base_difficulty

            def estimate_drilling_time(
                self,
                drill_points: List[Dict],
                drill_speed: str = "normal"
            ) -> Dict:
                """
                Estimate the time required for drilling through framing members.

                Args:
                    drill_points: List of drill points
                    drill_speed: Drilling speed ("slow", "normal", "fast")

                Returns:
                        Dict: Time estimates
                    """
                    if not drill_points:
                        return {
                            "total_time_minutes": 0,
                            "drill_points": 0,
                            "average_time_per_point": 0
                        }
    
                    # Base time per difficulty level (in minutes)
                    time_factors = {
                        "easy": {"slow": 5, "normal": 3, "fast": 2},
                        "moderate": {"slow": 8, "normal": 5, "fast": 3},
                        "difficult": {"slow": 12, "normal": 8, "fast": 5}
                    }
    
                    total_time = 0
    
                    for point in drill_points:
                        difficulty = point.get("difficulty", "moderate")
                        time_per_drill = time_factors.get(difficulty, time_factors["moderate"])[drill_speed]
                        total_time += time_per_drill
    
                    return {
                        "total_time_minutes": total_time,
                        "drill_points": len(drill_points),
                        "average_time_per_point": total_time / len(drill_points)
                    }
    
    def get_image_info(image_path: Union[str, Path]) -> Dict:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: Information about the image
            
        Raises:
            ImageProcessingError: If there is an error processing the image
        """
        from PIL import Image
        image_path = Path(image_path)
    
        raise ImageProcessingError(f"Error getting image info for {image_path}: {str(e)}")
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    info = {
                        'filename': image_path.name,
                        'path': str(image_path),
                        'format': img.format,
                        'mode': img.mode,
                        'width': img.width,
                        'height': img.height,
                        'size_bytes': image_path.stat().st_size,
                    }

                    # Try to get EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif() is not None:
                        info['has_exif'] = True
                    else:
                        info['has_exif'] = False

                    return info
            except Exception as e:
                # Using a generic exception since ImageProcessingError may not be defined
                raise Exception(f"Error getting image info for {image_path}: {str(e)}")

def list_images(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    List all valid images in a directory.

    Args:
        directory: Directory to search for images
        recursive: Whether to search recursively

    Returns:
        List[Path]: List of paths to valid images
    """
    import logging
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    valid_images = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return valid_images

    # Get all files
    if recursive:
        all_files = list(directory.glob('**/*'))
    else:
        all_files = list(directory.glob('*'))

    # Define valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    # Filter for image files
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            # Simple validation function (can be expanded)
            def validate_image(path):
                try:
                    from PIL import Image
                    with Image.open(path) as img:
                        return img.verify() is None
                except:
                    return False
            
            if validate_image(file_path):
                valid_images.append(file_path)

    return valid_images

# Example implementation for image collector 
class ImageCollector:
    """
    Class for collecting and organizing image datasets for training.
    """

    def __init__(self, target_dir=None):
        """
        Initialize the ImageCollector.

        Args:
            target_dir: Directory to store collected images
        """
        if target_dir is None:
            target_dir = Path('./data/images')
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_directory(self, source_dir,
                                 category="unclassified",
                                 recursive=True,
                                 copy=True,
                                 overwrite=False):
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
    def collect_from_directory(self, source_dir,
                                 category="unclassified",
                                 recursive=True,
                                 copy=True,
                                 overwrite=False):
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
            overwrite: Whether to overwrite existing files

        Returns:
            Dict: Statistics about the collection process
        """
        import logging
        logger = logging.getLogger(__name__)
        

            # Try to get EXIF data if available
            if hasattr(img, '_getexif') and img._getexif() is not None:
                info['has_exif'] = True
            else:
                info['has_exif'] = False

            return info
    except Exception as e:
        raise ImageProcessingError(f"Error getting image info for {image_path}: {str(e)}")

def list_images(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    List all valid images in a directory.

    Args:
        directory: Directory to search for images
        recursive: Whether to search recursively

    Returns:
        List[Path]: List of paths to valid images
    """
    from pathlib import Path
    import logging
    
    # Define valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    valid_images = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return valid_images

    # Get all files
    if recursive:
        all_files = list(directory.glob('**/*'))
    else:
        all_files = list(directory.glob('*'))

    # Filter for image files
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            # Use validate_image function which should be defined elsewhere
            if validate_image(file_path):
                valid_images.append(file_path)

    return valid_images

# The remaining code sections were causing errors and are commented out
"""
# data/collectors/image_collector.py


#Module for collecting and organizing images for the electrician time estimation application.


import os
import shutil
from pathlib import Path
from typing import Union, List, Optional, Dict
import uuid
import hashlib

from config.settings import IMAGES_DIR
from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import validate_image, get_image_info, list_images

logger = get_logger("image_collector")

class ImageCollector:
    """
    Class for collecting and organizing image datasets for training.
    """

    def __init__(self, target_dir: Union[str, Path] = IMAGES_DIR):
        """
        Initialize the ImageCollector.

        Args:
            target_dir: Directory to store collected images
        """
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_directory(self, source_dir: Union[str, Path],
                                 category: str = "unclassified",
                                 recursive: bool = True,
                                 copy: bool = True,
                                 overwrite: bool = False) -> Dict:
        """
        Collect images from a directory and store them in an organized manner.

        Args:
            source_dir: Directory to collect images from
            category: Category to assign to the images
            recursive: Whether to search recursively in source_dir
            copy: If True, copy files; if False, move files
            overwrite: Whether to overwrite existing files

        Returns:
            Dict: Statistics about the collection process
        """
        source_dir = Path(source_dir)

        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # Create category subdirectory if needed
        category_dir = self.target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Get list of valid images
        image_paths = list_images(source_dir, recursive=recursive)

        stats = {
            "total_found": len(image_paths),
            "successfully_collected": 0,
            "failed": 0,
            "skipped": 0
        }

        for img_path in image_paths:
            try:
                # Generate a unique filename based on content hash + original name
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:10]

                new_filename = f"{file_hash}_{img_path.name}"
                dest_path = category_dir / new_filename

                # Check if file already exists
                if dest_path.exists() and not overwrite:
                    logger.info(f"Skipping existing file: {dest_path}")
                    stats["skipped"] += 1
                    continue

                # Copy or move the file
                if copy:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(img_path, dest_path)

                stats["successfully_collected"] += 1
                logger.debug(f"Collected image: {dest_path}")

            except Exception as e:
                logger.error(f"Failed to collect image {img_path}: {str(e)}")
                stats["failed"] += 1

        logger.info(f"Collection completed. Stats: {stats}")
        return stats

    def collect_single_image(self, image_path: Union[str, Path],
                                  category: str = "unclassified",
                                  new_name: Optional[str] = None,
                                  copy: bool = True,
                                  overwrite: bool = False) -> Path:
        """
        Collect a single image and store it in an organized manner.

        Args:
            image_path: Path to the image file
            category: Category to assign to the image
            new_name: New name for the image (if None, generate a unique name)
            copy: If True, copy file; if False, move file
            overwrite: Whether to overwrite existing files

        Returns:
            Path: Path to the collected image
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        # Create category subdirectory if needed
        category_dir = self.target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if new_name is None:
            # Generate a unique filename based on content hash + original name
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:10]

            new_filename = f"{file_hash}_{image_path.name}"
        else:
            # Ensure the new name has the correct extension
            new_filename = f"{new_name}{image_path.suffix}"

        dest_path = category_dir / new_filename

        # Check if file already exists
        if dest_path.exists() and not overwrite:
            logger.info(f"File already exists: {dest_path}")
            return dest_path

        # Copy or move the file
        try:
            if copy:
                shutil.copy2(image_path, dest_path)
            else:
                shutil.move(image_path, dest_path)

            logger.info(f"Collected image: {dest_path}")
            return dest_path

        except Exception as e:
            raise ImageProcessingError(f"Failed to collect image {image_path}: {str(e)}")

    def create_dataset_index(self, output_file: Optional[Union[str, Path]] = None) -> Dict:
        """
        Create an index of all collected images.

        Args:
            output_file: Path to save the index (JSON format)

        Returns:
            Dict: Dataset index
        """
        index = {
            "total_images": 0,
            "categories": {}
        }

        # Scan through the target directory
        for category_dir in [d for d in self.target_dir.iterdir() if d.is_dir()]:
            category_name = category_dir.name
            image_paths = list_images(category_dir, recursive=False)

            category_data = {
                "count": len(image_paths),
                "images": []
            }

            for img_path in image_paths:
                try:
                    img_info = get_image_info(img_path)
                    category_data["images"].append({
                        "filename": img_path.name,
                        "path": str(img_path.relative_to(self.target_dir)),
                        "width": img_info.get('width'),
                        "height": img_info.get('height'),
                        "size_bytes": img_info.get('size_bytes')
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for {img_path}: {str(e)}")

            index["categories"][category_name] = category_data
            index["total_images"] += category_data["count"]

        # Save index to file if specified
        if output_file is not None:
            import json
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(index, f, indent=2)

            logger.info(f"Dataset index saved to {output_file}")

        return index
    
# Example implementation for image processor
# data/processors/image_processor.py
"""
Image processing utilities for the electrician time estimation application.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional, Any
import os

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("image_processor")

class ImageProcessor:
    """
    Class for processing images of framing members.
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImageProcessor.

        Args:
            target_size: Target size (width, height) for processed images
        """
        self.target_size = target_size

    def preprocess_image(self,
                           image: Union[str, Path, np.ndarray],
                           normalize: bool = True,
                           enhance_contrast: bool = False,
                           denoise: bool = False) -> np.ndarray:
        """
        Preprocess an image for analysis or model input.

        Args:
            image: Image file path or numpy array
            normalize: Whether to normalize pixel values to [0,1]
            enhance_contrast: Whether to enhance image contrast
            denoise: Whether to apply denoising

        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Apply denoising if requested
        if denoise:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Enhance contrast if requested
        if enhance_contrast:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Resize if target size is specified
        if self.target_size is not None:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0

        return img

    def batch_process(self,
                        image_paths: List[Union[str, Path]],
                        output_dir: Union[str, Path],
                        preprocessing_params: Dict[str, Any] = None) -> List[Path]:
        """
        Process a batch of images and save the results.

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save processed images
            preprocessing_params: Parameters for preprocessing

        Returns:
            List[Path]: Paths to processed images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if preprocessing_params is None:
            preprocessing_params = {
                'normalize': False, # Don't normalize for saved images
                'enhance_contrast': True,
                'denoise': True
            }

        processed_paths = []

        for img_path in image_paths:
            try:
                img_path = Path(img_path)

                if not validate_image(img_path):
                    logger.warning(f"Skipping invalid image: {img_path}")
                    continue

                # Process the image
                processed_img = self.preprocess_image(img_path, **preprocessing_params)

                # For saving, convert back to uint8 if normalized
                if preprocessing_params.get('normalize', False):
                    processed_img = (processed_img * 255).astype(np.uint8)

                # Save the processed image
                output_path = output_dir / img_path.name
                save_image(processed_img, output_path)
                processed_paths.append(output_path)

                logger.debug(f"Processed image: {output_path}")

            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {str(e)}")

        logger.info(f"Batch processing completed. Processed {len(processed_paths)} images.")
        return processed_paths

    def extract_features(self,
                         image: Union[str, Path, np.ndarray],
                         feature_type: str = 'edges') -> np.ndarray:
        """
        Extract features from an image.

        Args:
            image: Image file path or numpy array
            feature_type: Type of features to extract ('edges', 'corners', 'lines')

        Returns:
            np.ndarray: Feature image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image, color_mode='grayscale')
        elif len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()

        if feature_type == 'edges':
            # Edge detection (good for framing members)
            features = cv2.Canny(img, 50, 150)
        elif feature_type == 'corners':
            # Corner detection
            corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
            features = np.zeros_like(img)
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(features, (int(x), int(y)), 3, 255, -1)
        elif feature_type == 'lines':
            # Line detection (good for framing members)
            edges = cv2.Canny(img, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            features = np.zeros_like(img)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(features, (x1, y1), (x2, y2), 255, 2)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return features

    def analyze_framing(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """
        Basic analysis of framing members in an image.

        Args:
            image: Image file path or numpy array

        Returns:
            Dict: Analysis results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
            original_img = img.copy()
        else:
            img = image.copy()
            original_img = img.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        results = {
            "has_framing_lines": lines is not None and len(lines) > 0,
            "line_count": 0 if lines is None else len(lines),
            "orientation_stats": {"horizontal": 0, "vertical": 0, "diagonal": 0},
            "visualization": None
        }

        if lines is not None:
            # Create visualization image
            viz_img = original_img.copy()

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle to determine orientation
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                # Classify line orientation
                if angle < 20 or angle > 160:
                    results["orientation_stats"]["horizontal"] += 1
                    color = (0, 255, 0) # Green for horizontal
                elif angle > 70 and angle < 110:
                    results["orientation_stats"]["vertical"] += 1
                    color = (255, 0, 0) # Red for vertical
                else:
                    results["orientation_stats"]["diagonal"] += 1
                    color = (0, 0, 255) # Blue for diagonal

                # Draw line on visualization
                cv2.line(viz_img, (x1, y1), (x2, y2), color, 2)

            # Store visualization
            results["visualization"] = viz_img

        return results

# data/processors/augmentation.py
"""
Image augmentation for the electrician time estimation application.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import os
import json

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("augmentation")

class ImageAugmenter:
    """
    Class for augmenting images of framing members for training.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the ImageAugmenter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

        # Define transformations suitable for residential framing images
        self.basic_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
        ])

        # More aggressive transformations
        self.advanced_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.GaussianBlur(blur_limit=(3, 9), p=0.4),
            A.GaussNoise(var_limit=(10, 80), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.8),
            A.RandomShadow(p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
        ])

        # Special transformations for lighting simulation
        self.lighting_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=0.7),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.8),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
        ])

    def augment_image(self,
                      image: Union[str, Path, np.ndarray],
                      transform_type: str = 'basic',
                      save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            image: Image file path or numpy array
            transform_type: Type of transformation ('basic', 'advanced', 'lighting')
            save_path: Path to save the augmented image

        Returns:
            np.ndarray: Augmented image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Select transformation based on type
        if transform_type == 'basic':
            transform = self.basic_transforms
        elif transform_type == 'advanced':
            transform = self.advanced_transforms
        elif transform_type == 'lighting':
            transform = self.lighting_transforms
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")

        # Apply transformation
        augmented = transform(image=img)['image']

        # Save if path is provided
        if save_path is not None:
            save_image(augmented, save_path)

        return augmented

    def create_augmentation_set(self,
                                 image_paths: List[Union[str, Path]],
                                 output_dir: Union[str, Path],
                                 transform_types: List[str] = ['basic'],
                                 samples_per_image: int = 3,
                                 include_original: bool = True) -> Dict:
        """
        Create a set of augmented images from a list of original images.

        Args:
            image_paths: List of image paths to augment
            output_dir: Directory to save augmented images
            transform_types: Types of transformations to apply
            samples_per_image: Number of augmented samples to generate per image
            include_original: Whether to include original images in the output

        Returns:
            Dict: Statistics about the augmentation process
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "original_images": len(image_paths),
            "augmented_images": 0,
            "total_images": 0,
            "transform_counts": {t: 0 for t in transform_types}
        }

        # Process each image
        for img_path in image_paths:
            try:
                img_path = Path(img_path)

                if not validate_image(img_path):
                    logger.warning(f"Skipping invalid image: {img_path}")
                    continue

                # Include original if requested
                if include_original:
                    orig_save_path = output_dir / f"orig_{img_path.name}"
                    img = load_image(img_path)
                    save_image(img, orig_save_path)
                    stats["total_images"] += 1

                # Generate augmented samples
                for i in range(samples_per_image):
                    for transform_type in transform_types:
                        try:
                            # Create unique filename for augmented image
                            aug_filename = f"aug_{transform_type}_{i}_{img_path.name}"
                            aug_save_path = output_dir / aug_filename

                            # Apply augmentation and save
                            self.augment_image(img_path, transform_type=transform_type, save_path=aug_save_path)

                            stats["augmented_images"] += 1
                            stats["transform_counts"][transform_type] += 1
                            stats["total_images"] += 1

                        except Exception as e:
                            logger.error(f"Failed to augment image {img_path} with {transform_type}: {str(e)}")

            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {str(e)}")

        # Save statistics to file
        stats_file = output_dir / "augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Augmentation completed. Created {stats['augmented_images']} augmented images.")
        return stats

    def create_custom_transform(self,
                                  transform_config: Dict[str, Any]) -> A.Compose:
        """
        Create a custom augmentation transform from a configuration.

        Args:
            transform_config: Dictionary of transform parameters

        Returns:
            A.Compose: Custom augmentation pipeline
        """
        transforms = []

        # Parse configuration and create transforms
        for transform_name, params in transform_config.items():
            if transform_name == "RandomBrightnessContrast":
                transforms.append(A.RandomBrightnessContrast(**params))
            elif transform_name == "GaussianBlur":
                transforms.append(A.GaussianBlur(**params))
            elif transform_name == "GaussNoise":
                transforms.append(A.GaussNoise(**params))
            elif transform_name == "HorizontalFlip":
                transforms.append(A.HorizontalFlip(**params))
            elif transform_name == "VerticalFlip":
                transforms.append(A.VerticalFlip(**params))
            elif transform_name == "ShiftScaleRotate":
                transforms.append(A.ShiftScaleRotate(**params))
            elif transform_name == "RandomShadow":
                transforms.append(A.RandomShadow(**params))
            elif transform_name == "RandomFog":
                transforms.append(A.RandomFog(**params))
            elif transform_name == "CLAHE":
                transforms.append(A.CLAHE(**params))
            elif transform_name == "HueSaturationValue":
                transforms.append(A.HueSaturationValue(**params))
            else:
                logger.warning(f"Unsupported transform: {transform_name}")

        return A.Compose(transforms)

    def augment_with_custom_transform(self,
                                       image: Union[str, Path, np.ndarray],
                                       custom_transform: A.Compose,
                                       save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Apply a custom augmentation transform to an image.

        Args:
            image: Image file path or numpy array
            custom_transform: Custom augmentation transform
            save_path: Path to save the augmented image

        Returns:
            np.ndarray: Augmented image
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image.copy()

        # Apply transformation
        augmented = custom_transform(image=img)['image']

        # Save if path is provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(augmented, save_path)

        return augmented



# data/annotation/annotation_tools.py
"""
Annotation tools for labeling framing members in images.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os
from typing import Union, List, Dict, Optional, Tuple
from datetime import datetime
import uuid

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image

logger = get_logger("annotation_tools")

class AnnotationTool:
    """
    Class for creating and managing annotations for framing member images.
    """

    # Define categories for framing members
    CATEGORIES = [
        {"id": 1, "name": "stud", "supercategory": "framing"},
        {"id": 2, "name": "joist", "supercategory": "framing"},
        {"id": 3, "name": "rafter", "supercategory": "framing"},
        {"id": 4, "name": "beam", "supercategory": "framing"},
        {"id": 5, "name": "plate", "supercategory": "framing"},
        {"id": 6, "name": "obstacle", "supercategory": "obstacle"},
        {"id": 7, "name": "electrical_box", "supercategory": "electrical"},
        {"id": 8, "name": "plumbing", "supercategory": "obstacle"},
    ]

    def __init__(self, annotation_dir: Union[str, Path]):
        """
        Initialize the AnnotationTool.

        Args:
            annotation_dir: Directory to store annotations
        """
        self.annotation_dir = Path(annotation_dir)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        # Create a category lookup for faster access
        self.category_lookup = {cat["id"]: cat for cat in self.CATEGORIES}

    def create_coco_annotation(self,
                                image_path: Union[str, Path],
                                annotations: List[Dict],
                                output_file: Optional[Union[str, Path]] = None) -> Dict:
        """
        Create a COCO format annotation for an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the COCO JSON file

        Returns:
            Dict: COCO format annotation
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        try:
            # Load image to get dimensions
            img = load_image(image_path)
            height, width = img.shape[:2]

            # Create image info
            image_id = int(uuid.uuid4().int % (2**31 - 1)) # Random positive 32-bit int
            image_info = {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Create annotation objects
            coco_annotations = []
            for i, anno in enumerate(annotations):
                # Validate category_id
                category_id = anno.get("category_id")
                if category_id not in self.category_lookup:
                    logger.warning(f"Invalid category_id: {category_id}, skipping annotation")
                    continue

                # Get bbox
                bbox = anno.get("bbox")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}, skipping annotation")
                    continue

                # Calculate segmentation from bbox
                x, y, w, h = bbox
                segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]

                # Calculate area
                area = w * h

                # Create annotation object
                annotation_obj = {
                    "id": int(uuid.uuid4().int % (2**31 - 1)),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0
                }

                coco_annotations.append(annotation_obj)

            # Create full COCO dataset structure
            coco_data = {
                "info": {
                    "description": "Electrician Time Estimator Dataset",
                    "url": "",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "contributor": "Electrician Time Estimator",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Attribution-NonCommercial",
                        "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                    }
                ],
                "categories": self.CATEGORIES,
                "images": [image_info],
                "annotations": coco_annotations
            }

            # Save to file if specified
            if output_file is not None:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    json.dump(coco_data, f, indent=2)

                logger.info(f"COCO annotation saved to {output_file}")

            return coco_data

        except Exception as e:
            raise ImageProcessingError(f"Failed to create COCO annotation for {image_path}: {str(e)}")

    def create_yolo_annotation(self,
                                image_path: Union[str, Path],
                                annotations: List[Dict],
                                output_file: Optional[Union[str, Path]] = None) -> List[str]:
        """
        Create a YOLO format annotation for an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the YOLO txt file

        Returns:
            List[str]: YOLO format annotation lines
        """
        image_path = Path(image_path)

        if not validate_image(image_path):
            raise ImageProcessingError(f"Invalid or corrupted image: {image_path}")

        try:
            # Load image to get dimensions
            img = load_image(image_path)
            img_height, img_width = img.shape[:2]

            # Create YOLO annotation lines
            yolo_lines = []

            for anno in annotations:
                # Validate category_id
                category_id = anno.get("category_id")
                if category_id not in self.category_lookup:
                    logger.warning(f"Invalid category_id: {category_id}, skipping annotation")
                    continue

                # YOLO uses 0-indexed class numbers
                yolo_class = category_id - 1

                # Get bbox in COCO format [x, y, width, height]
                bbox = anno.get("bbox")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}, skipping annotation")
                    continue

                # Convert COCO bbox to YOLO format (normalized center x, center y, width, height)
                x, y, w, h = bbox
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                # Create YOLO line
                yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)

            # Save to file if specified
            if output_file is not None:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                logger.info(f"YOLO annotation saved to {output_file}")

            return yolo_lines

        except Exception as e:
            raise ImageProcessingError(f"Failed to create YOLO annotation for {image_path}: {str(e)}")

    def visualize_annotations(self,
                                  image_path: Union[str, Path],
                                  annotations: List[Dict],
                                  output_file: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize annotations on an image.

        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with format:
                [{"category_id": int, "bbox": [x, y, width, height]}, ...]
            output_file: Path to save the visualization

        Returns:
            np.ndarray: Image with visualized annotations
        """
        image_path = Path(image_path)

        try:
            # Load image
            img = load_image(image_path)
            vis_img = img.copy()

            # Define colors for each category (BGR format for OpenCV)
            colors = [
                (0, 255, 0), # stud (green)
                (255, 0, 0), # joist (blue)
                (0, 0, 255), # rafter (red)
                (255, 255, 0), # beam (cyan)
                (255, 0, 255), # plate (magenta)
                (0, 255, 255), # obstacle (yellow)
                (128, 0, 128), # electrical_box (purple)
                (0, 128, 128), # plumbing (brown)
            ]

            # Draw each annotation
            for anno in annotations:
                category_id = anno.get("category_id")
                bbox = anno.get("bbox")

                if category_id not in self.category_lookup or not bbox or len(bbox) != 4:
                    continue

                # Get category name and color
                category_name = self.category_lookup[category_id]["name"]
                color = colors[(category_id - 1) % len(colors)]

                # Draw bounding box
                x, y, w, h = [int(c) for c in bbox]
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)

                # Draw label background
                text_size = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_img, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)

                # Draw label text
                cv2.putText(vis_img, category_name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save visualization if specified
            if output_file is not None:
                save_image(vis_img, output_file)
                logger.info(f"Visualization saved to {output_file}")

            return vis_img

        except Exception as e:
            raise ImageProcessingError(f"Failed to visualize annotations for {image_path}: {str(e)}")

    def merge_coco_annotations(self,
                                 annotation_files: List[Union[str, Path]],
                                 output_file: Union[str, Path]) -> Dict:
        """
        Merge multiple COCO annotation files into a single dataset.

        Args:
            annotation_files: List of COCO annotation files to merge
            output_file: Path to save the merged annotation file

        Returns:
            Dict: Merged COCO dataset
        """
        merged_dataset = {
            "info": {
                "description": "Merged Electrician Time Estimator Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Electrician Time Estimator",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }
            ],
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        # Track image and annotation IDs to avoid duplicates
        image_ids = set()
        anno_ids = set()

        # Process each annotation file
        for anno_file in annotation_files:
            try:
                with open(anno_file, 'r') as f:
                    dataset = json.load(f)

                # Add images (avoid duplicates by ID)
                for img in dataset.get("images", []):
                    if img["id"] not in image_ids:
                        merged_dataset["images"].append(img)
                        image_ids.add(img["id"])

                # Add annotations (avoid duplicates by ID)
                for anno in dataset.get("annotations", []):
                    if anno["id"] not in anno_ids:
                        merged_dataset["annotations"].append(anno)
                        anno_ids.add(anno["id"])

            except Exception as e:
                logger.error(f"Failed to process annotation file {anno_file}: {str(e)}")

        # Save merged dataset
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(merged_dataset, f, indent=2)

        logger.info(f"Merged {len(annotation_files)} annotation files to {output_file}")
        logger.info(f"Merged dataset has {len(merged_dataset['images'])} images and {len(merged_dataset['annotations'])} annotations")

        return merged_dataset


# data/annotation/annotation_converter.py
"""
Utilities for converting between annotation formats.
"""

import os
import json
import numpy as np
from pathlib import Path
import cv2
from typing import Union, List, Dict, Optional, Tuple
import glob

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, validate_image

logger = get_logger("annotation_converter")

class AnnotationConverter:
    """
    Class for converting between different annotation formats.
    """

    def __init__(self, categories: Optional[List[Dict]] = None):
        """
        Initialize the AnnotationConverter.

        Args:
            categories: List of category dictionaries (optional)
                Format: [{"id": int, "name": str, "supercategory": str}, ...]
        """
        # Default categories if none provided
        self.categories = categories or [
            {"id": 1, "name": "stud", "supercategory": "framing"},
            {"id": 2, "name": "joist", "supercategory": "framing"},
            {"id": 3, "name": "rafter", "supercategory": "framing"},
            {"id": 4, "name": "beam", "supercategory": "framing"},
            {"id": 5, "name": "plate", "supercategory": "framing"},
            {"id": 6, "name": "obstacle", "supercategory": "obstacle"},
            {"id": 7, "name": "electrical_box", "supercategory": "electrical"},
            {"id": 8, "name": "plumbing", "supercategory": "obstacle"},
        ]

        # Create category lookups
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self.name_to_id = {cat["name"]: cat["id"] for cat in self.categories}

    def yolo_to_coco(self,
                     yolo_dir: Union[str, Path],
                     image_dir: Union[str, Path],
                     output_file: Union[str, Path],
                     image_ext: str = ".jpg") -> Dict:
        """
        Convert YOLO format annotations to COCO format.

        Args:
            yolo_dir: Directory containing YOLO annotation files
            image_dir: Directory containing corresponding images
            output_file: Path to save the COCO JSON file
            image_ext: Image file extension

        Returns:
            Dict: COCO format dataset
        """
        yolo_dir = Path(yolo_dir)
        image_dir = Path(image_dir)

        # Initialize COCO dataset structure
        coco_data = {
            "info": {
                "description": "Converted from YOLO format",
                "url": "",
                "version": "1.0",
                "year": 2023,
                "contributor": "Electrician Time Estimator",
                "date_created": ""
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc/2.0/"
                }
            ],
            "categories": self.categories,
            "images": [],
            "annotations": []
        }

        # Find all YOLO annotation files
        yolo_files = list(yolo_dir.glob("*.txt"))

        # Initialize counters for IDs
        image_id = 0
        annotation_id = 0

        # Process each YOLO file
        for yolo_file in yolo_files:
            try:
                # Determine corresponding image path
                image_stem = yolo_file.stem
                image_path = image_dir / f"{image_stem}{image_ext}"

                if not image_path.exists():
                    # Try finding the image with different extensions
                    potential_images = list(image_dir.glob(f"{image_stem}.*"))
                    if potential_images:
                        image_path = potential_images[0]
                    else:
                        logger.warning(f"Image not found for annotation: {yolo_file}")
                        continue

                if not validate_image(image_path):
                    logger.warning(f"Invalid image file: {image_path}")
                    continue

                # Load image to get dimensions
                img = load_image(image_path)
                height, width = img.shape[:2]

                # Add image info to COCO dataset
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "date_captured": ""
                })

                # Read YOLO annotations
                with open(yolo_file, 'r') as f:
                    yolo_annotations = f.readlines()

                # Convert each YOLO annotation line to COCO format
                for line in yolo_annotations:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse YOLO line: class x_center y_center width height
                        elements = line.split()
                        if len(elements) != 5:
                            logger.warning(f"Invalid YOLO annotation format: {line}")
                            continue

                        class_id, x_center, y_center, bbox_width, bbox_height = elements

                        # Convert to float
                        class_id = int(class_id)
                        x_center = float(x_center)
                        y_center = float(y_center)
                        bbox_width = float(bbox_width)
                        bbox_height = float(bbox_height)

                        # YOLO coordinates are normalized, convert to absolute
                        abs_width = bbox_width * width
                        abs_height = bbox_height * height
                        abs_x = (x_center * width) - (abs_width / 2)
                        abs_y = (y_center * height) - (abs_height / 2)

                        # YOLO classes are 0-indexed, COCO uses the category ID
                        coco_category_id = class_id + 1

                        # Create segmentation from bbox (simple polygon)
                        segmentation = [
                            [
                                abs_x, abs_y,
                                abs_x + abs_width, abs_y,
                                abs_x + abs_width, abs_y + abs_height,
                                abs_x, abs_y + abs_height
                            ]
                        ]

                        # Add annotation to COCO dataset
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": coco_category_id,
                            "bbox": [abs_x, abs_y, abs_width, abs_height],
                            "area": abs_width * abs_height,
                            "segmentation": segmentation,
                            "iscrowd": 0
                        })

                        annotation_id += 1

                    except Exception as e:
                        logger.warning(f"Error processing annotation line: {line}. Error: {str(e)}")

                # Increment image ID for next file
                image_id += 1

            except Exception as e:
                logger.error(f"Failed to process YOLO file {yolo_file}: {str(e)}")

        # Save COCO dataset to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"Converted {len(yolo_files)} YOLO files to COCO format: {output_file}")
        logger.info(f"Dataset has {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

        return coco_data

    def coco_to_yolo(self,
                     coco_file: Union[str, Path],
                     output_dir: Union[str, Path]) -> Dict:
        """
        Convert COCO format annotations to YOLO format.

        Args:
            coco_file: Path to COCO JSON file
            output_dir: Directory to save YOLO annotation files

        Returns:
            Dict: Statistics about the conversion
        """
        coco_file = Path(coco_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load COCO dataset
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)

            # Create lookup dict for images
            images = {img["id"]: img for img in coco_data["images"]}

            # Track statistics
            stats = {
                "total_images": len(coco_data["images"]),
                "total_annotations": len(coco_data["annotations"]),
                "converted_images": 0,
                "converted_annotations": 0
            }

            # Group annotations by image_id
            annotations_by_image = {}
            for anno in coco_data["annotations"]:
                image_id = anno["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(anno)

            # Process each image
            for image_id, image_info in images.items():
                # Get all annotations for this image
                image_annotations = annotations_by_image.get(image_id, [])

                if not image_annotations:
                    continue

                # Get image dimensions
                img_width = image_info["width"]
                img_height = image_info["height"]

                # Create YOLO file path
                yolo_file = output_dir / f"{Path(image_info['file_name']).stem}.txt"

                # Convert annotations to YOLO format
                yolo_lines = []

                for anno in image_annotations:
                    try:
                        # Get category ID (COCO) and convert to class ID (YOLO, 0-indexed)
                        coco_category_id = anno["category_id"]
                        yolo_class_id = coco_category_id - 1

                        # Get bounding box
                        x, y, width, height = anno["bbox"]

                        # Convert to YOLO format (normalized center x, center y, width, height)
                        x_center = (x + width / 2) / img_width
                        y_center = (y + height / 2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height

                        # Create YOLO line
                        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        yolo_lines.append(yolo_line)

                        stats["converted_annotations"] += 1

                    except Exception as e:
                        logger.warning(f"Error converting annotation: {str(e)}")

                # Write YOLO file
                with open(yolo_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                stats["converted_images"] += 1

            logger.info(f"Converted COCO to YOLO format: {stats}")
            return stats

        except Exception as e:
            raise ValueError(f"Failed to convert COCO to YOLO: {str(e)}")


# data/pipeline/data_pipeline.py
"""
Data pipeline for processing images for the electrician time estimation application.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import json
import numpy as np
import cv2
from tqdm import tqdm

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import load_image, save_image, validate_image, list_images
from data.processors.image_processor import ImageProcessor
from data.processors.augmentation import ImageAugmenter

logger = get_logger("data_pipeline")

class DataPipeline:
    """
    Pipeline for processing and preparing images for model training.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataPipeline.

        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or {}

        # Initialize processors
        self.image_processor = ImageProcessor(
            target_size=self.config.get("target_size")
        )
        self.augmenter = ImageAugmenter(
            seed=self.config.get("seed", 42)
        )

    def process_dataset(self,
                           input_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           annotation_dir: Optional[Union[str, Path]] = None,
                           preprocessing: Optional[Dict[str, Any]] = None,
                           augmentation: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Process a dataset of images and annotations.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            annotation_dir: Directory containing annotations (optional)
            preprocessing: Preprocessing parameters
            augmentation: Augmentation parameters

        Returns:
            Dict: Statistics about the processing
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy annotations if provided
        if annotation_dir is not None:
            annotation_dir = Path(annotation_dir)
            output_anno_dir = output_dir / "annotations"
            output_anno_dir.mkdir(parents=True, exist_ok=True)

            # Copy all annotation files
            for anno_file in annotation_dir.glob("*"):
                if anno_file.is_file():
                    shutil.copy2(anno_file, output_anno_dir / anno_file.name)

        # Set default preprocessing parameters
        if preprocessing is None:
            preprocessing = {
                "normalize": False,
                "enhance_contrast": True,
                "denoise": True
            }

        # Set default augmentation parameters
        if augmentation is None:
            augmentation = {
                "enabled": False,
                "transform_types": ["basic"],
                "samples_per_image": 2
            }

        # Find all valid images
        image_paths = list_images(input_dir)

        # Create processed images directory
        processed_dir = output_dir / "images"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create augmented images directory if augmentation is enabled
        if augmentation.get("enabled", False):
            augmented_dir = output_dir / "augmented"
            augmented_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        stats = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "augmented_images": 0,
            "failed_images": 0
        }

        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Preprocess image
                processed_img = self.image_processor.preprocess_image(
                    img_path,
                    normalize=preprocessing.get("normalize", False),
                    enhance_contrast=preprocessing.get("enhance_contrast", True),
                    denoise=preprocessing.get("denoise", True)
                )

                # For saving, convert back to uint8 if normalized
                if preprocessing.get("normalize", False):
                    processed_img = (processed_img * 255).astype(np.uint8)

                # Save processed image
                processed_path = processed_dir / img_path.name
                save_image(processed_img, processed_path)
                stats["processed_images"] += 1

                # Perform augmentation if enabled
                if augmentation.get("enabled", False):
                    aug_samples = augmentation.get("samples_per_image", 2)
                    transform_types = augmentation.get("transform_types", ["basic"])

                    for i in range(aug_samples):
                        for transform_type in transform_types:
                            try:
                                # Create unique filename for augmented image
                                aug_filename = f"aug_{transform_type}_{i}_{img_path.name}"
                                aug_path = augmented_dir / aug_filename

                                # Apply augmentation and save
                                self.augmenter.augment_image(
                                    processed_img,
                                    transform_type=transform_type,
                                    save_path=aug_path
                                )

                                stats["augmented_images"] += 1

                            except Exception as e:
                                logger.error(f"Augmentation failed for {img_path} with {transform_type}: {str(e)}")

            except Exception as e:
                logger.error(f"Processing failed for {img_path}: {str(e)}")
                stats["failed_images"] += 1

        # Save processing statistics
        stats_file = output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset processing completed. Stats: {stats}")
        return stats

    def prepare_model_inputs(self,
                              image_paths: List[Union[str, Path]],
                              target_size: Optional[Tuple[int, int]] = None,
                              normalize: bool = True,
                              batch_size: int = 32) -> np.ndarray:
        """
        Prepare images as input for the model.

        Args:
            image_paths: List of image paths
            target_size: Target size for images (width, height)
            normalize: Whether to normalize pixel values
            batch_size: Batch size for processing

        Returns:
            np.ndarray: Batch of processed images
        """
        # Override target_size if provided
        if target_size is not None:
            self.image_processor = ImageProcessor(target_size=target_size)

        # Initialize batch array
        if target_size:
            width, height = target_size
        else:
            # Load the first image to get dimensions
            first_img = load_image(image_paths[0])
            height, width = first_img.shape[:2]

        # Determine shape based on normalization
        if normalize:
            batch = np.zeros((len(image_paths), height, width, 3), dtype=np.float32)
        else:
            batch = np.zeros((len(image_paths), height, width, 3), dtype=np.uint8)

        # Process each image
        for i, img_path in enumerate(image_paths):
            try:
                processed_img = self.image_processor.preprocess_image(
                    img_path,
                    normalize=normalize,
                    enhance_contrast=True,
                    denoise=True
                )

                batch[i] = processed_img

            except Exception as e:
                logger.error(f"Failed to prepare image {img_path}: {str(e)}")
                # Fill with zeros for failed images

        return batch


# data/pipeline/dataset_splitter.py
"""
Utilities for splitting datasets into training, validation, and test sets.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
from sklearn.model_selection import train_test_split

from utils.exceptions import ImageProcessingError
from utils.logger import get_logger
from data.utils.image_utils import list_images

logger = get_logger("dataset_splitter")

class DatasetSplitter:
    """
    Class for splitting datasets into training, validation, and test sets.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the DatasetSplitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def split_dataset(self,
                      dataset_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                      copy_files: bool = True,
                      annotation_dir: Optional[Union[str, Path]] = None,
                      annotation_ext: str = ".txt") -> Dict:
        """
        Split a dataset into training, validation, and test sets.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)
            annotation_dir: Directory containing annotations (optional)
            annotation_ext: Extension of annotation files

        Returns:
            Dict: Statistics about the splitting process
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        # Validate split ratios
        if sum(split_ratios) != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        # Create output directories
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create 'images' subdirectory
        (d / "images").mkdir(exist_ok=True)

        # Create 'annotations' subdirectory if annotation_dir is provided
        if annotation_dir is not None:
            (d / "annotations").mkdir(exist_ok=True)

        # List all valid images
        image_paths = list_images(dataset_dir)

        if not image_paths:
            raise ValueError(f"No valid images found in {dataset_dir}")

        # Split the dataset
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_paths, temp_paths = train_test_split(
            image_paths,
            train_size=train_ratio,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths,
            train_size=val_ratio_adjusted,
            random_state=self.seed
        )

        # Track statistics
        stats = {
            "total_images": len(image_paths),
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "test_images": len(test_paths),
            "train_annotations": 0,
            "val_annotations": 0,
            "test_annotations": 0
        }

        # Helper function to copy/move files
        def transfer_files(paths, target_dir, file_type="images"):
            count = 0
            for src_path in paths:
                # Determine destination path
                if file_type == "images":
                    dst_path = target_dir / "images" / src_path.name
                else: # annotations
                    # Use the same filename as the image but with annotation extension
                    dst_path = target_dir / "annotations" / f"{src_path.stem}{annotation_ext}"

                # Skip if destination exists and is not empty
                if dst_path.exists() and dst_path.stat().st_size > 0:
                    count += 1
                    continue

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer {src_path} to {dst_path}: {str(e)}")

            return count

        # Transfer images
        transfer_files(train_paths, train_dir)
        transfer_files(val_paths, val_dir)
        transfer_files(test_paths, test_dir)

        # Transfer annotations if annotation_dir is provided
        if annotation_dir is not None:
            annotation_dir = Path(annotation_dir)

            # Helper function to find annotation file for an image
            def find_annotation(image_path):
                anno_path = annotation_dir / f"{image_path.stem}{annotation_ext}"
                if anno_path.exists():
                    return anno_path
                return None

            # Get annotation paths for each set
            train_anno_paths = [find_annotation(img) for img in train_paths]
            val_anno_paths = [find_annotation(img) for img in val_paths]
            test_anno_paths = [find_annotation(img) for img in test_paths]

            # Remove None values
            train_anno_paths = [p for p in train_anno_paths if p is not None]
            val_anno_paths = [p for p in val_anno_paths if p is not None]
            test_anno_paths = [p for p in test_anno_paths if p is not None]

            # Transfer annotations
            stats["train_annotations"] = transfer_files(train_anno_paths, train_dir, "annotations")
            stats["val_annotations"] = transfer_files(val_anno_paths, val_dir, "annotations")
            stats["test_annotations"] = transfer_files(test_anno_paths, test_dir, "annotations")

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset split completed. Stats: {stats}")
        return stats

    def create_stratified_split(self,
                                  dataset_dir: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  annotation_dir: Union[str, Path],
                                  split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                                  copy_files: bool = True) -> Dict:
        """
        Create a stratified split based on annotation categories.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing annotations in COCO or YOLO format
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        annotation_dir = Path(annotation_dir)

        # Determine annotation format (COCO or YOLO)
        coco_files = list(annotation_dir.glob("*.json"))
        is_coco = len(coco_files) > 0

        if is_coco:
            # Process COCO format
            return self._stratified_split_coco(
                dataset_dir, output_dir, annotation_dir, split_ratios, copy_files
            )
        else:
            # Process YOLO format
            return self._stratified_split_yolo(
                dataset_dir, output_dir, annotation_dir, split_ratios, copy_files
            )

    def _stratified_split_coco(self,
                                   dataset_dir: Path,
                                   output_dir: Path,
                                   annotation_dir: Path,
                                   split_ratios: Tuple[float, float, float],
                                   copy_files: bool) -> Dict:
        """
        Create a stratified split for COCO format annotations.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing COCO format annotations
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        # Find COCO annotation file
        coco_files = list(annotation_dir.glob("*.json"))
        if not coco_files:
            raise ValueError("No COCO annotation files found")

        coco_file = coco_files[0]

        # Load COCO dataset
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Create category statistics
        category_images = {}
        for annotation in coco_data.get("annotations", []):
            category_id = annotation["category_id"]
            image_id = annotation["image_id"]

            if category_id not in category_images:
                category_images[category_id] = set()

            category_images[category_id].add(image_id)

        # Get image IDs with their highest frequency category
        image_category = {}
        for category_id, image_ids in category_images.items():
            for image_id in image_ids:
                if image_id not in image_category:
                    image_category[image_id] = []
                image_category[image_id].append(category_id)

        # Create stratification labels (use the first category for simplicity)
        stratify_labels = []
        image_ids = []

        for image_id, categories in image_category.items():
            image_ids.append(image_id)
            stratify_labels.append(categories[0])

        # Split while preserving category distribution
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_ids, temp_ids, _, temp_labels = train_test_split(
            image_ids,
            stratify_labels,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.seed
        )

        # Create image ID to filename mapping
        id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

        # Create output directories
        train_dir = output_dir / "train" / "images"
        val_dir = output_dir / "val" / "images"
        test_dir = output_dir / "test" / "images"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Helper function to transfer files
        def transfer_images(image_ids, target_dir):
            count = 0
            for image_id in image_ids:
                filename = id_to_filename.get(image_id)
                if not filename:
                    logger.warning(f"No filename found for image ID {image_id}")
                    continue

                src_path = dataset_dir / filename
                dst_path = target_dir / filename

                if not src_path.exists():
                    logger.warning(f"Source image not found: {src_path}")
                    continue

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer {src_path}: {str(e)}")

            return count

        # Transfer images
        train_count = transfer_images(train_ids, train_dir)
        val_count = transfer_images(val_ids, val_dir)
        test_count = transfer_images(test_ids, test_dir)

        # Create COCO annotation files for each split
        def create_split_coco(image_ids, output_file):
            split_coco = {
                "info": coco_data.get("info", {}),
                "licenses": coco_data.get("licenses", []),
                "categories": coco_data.get("categories", []),
                "images": [],
                "annotations": []
            }

            # Add images
            for img in coco_data.get("images", []):
                if img["id"] in image_ids:
                    split_coco["images"].append(img)

            # Add annotations
            for anno in coco_data.get("annotations", []):
                if anno["image_id"] in image_ids:
                    split_coco["annotations"].append(anno)

            # Save to file
            with open(output_file, 'w') as f:
                json.dump(split_coco, f, indent=2)

            return len(split_coco["annotations"])

        # Create annotation directories
        train_anno_dir = output_dir / "train" / "annotations"
        val_anno_dir = output_dir / "val" / "annotations"
        test_anno_dir = output_dir / "test" / "annotations"

        for d in [train_anno_dir, val_anno_dir, test_anno_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create split annotations
        train_anno_count = create_split_coco(train_ids, train_anno_dir / "instances.json")
        val_anno_count = create_split_coco(val_ids, val_anno_dir / "instances.json")
        test_anno_count = create_split_coco(test_ids, test_anno_dir / "instances.json")

        # Create statistics
        stats = {
            "total_images": len(image_ids),
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "train_annotations": train_anno_count,
            "val_annotations": val_anno_count,
            "test_annotations": test_anno_count,
            "category_distribution": {
                str(cat_id): len(img_ids) for cat_id, img_ids in category_images.items()
            }
        }

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stratified split completed. Stats: {stats}")
        return stats

    def _stratified_split_yolo(self,
                                  dataset_dir: Path,
                                  output_dir: Path,
                                  annotation_dir: Path,
                                  split_ratios: Tuple[float, float, float],
                                  copy_files: bool) -> Dict:
        """
        Create a stratified split for YOLO format annotations.

        Args:
            dataset_dir: Directory containing the dataset images
            output_dir: Directory to save the split datasets
            annotation_dir: Directory containing YOLO format annotations
            split_ratios: Ratios for train, validation, and test sets
            copy_files: Whether to copy files (True) or move them (False)

        Returns:
            Dict: Statistics about the splitting process
        """
        # Find all YOLO annotation files
        yolo_files = list(annotation_dir.glob("*.txt"))

        if not yolo_files:
            raise ValueError("No YOLO annotation files found")

        # Parse annotations to get category distribution
        image_categories = {}

        for yolo_file in yolo_files:
            try:
                with open(yolo_file, 'r') as f:
                    lines = f.readlines()

                categories = set()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        categories.add(class_id)

                if categories:
                    image_categories[yolo_file.stem] = list(categories)

            except Exception as e:
                logger.warning(f"Failed to parse YOLO file {yolo_file}: {str(e)}")

        # Create stratification labels (use the first category for simplicity)
        image_names = list(image_categories.keys())
        stratify_labels = [cats[0] for cats in image_categories.values()]

        # Split while preserving category distribution
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split into train and temp (val+test)
        train_names, temp_names, _, temp_labels = train_test_split(
            image_names,
            stratify_labels,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=self.seed
        )

        # Then split temp into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_names, test_names = train_test_split(
            temp_names,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.seed
        )

        # Create output directories
        train_img_dir = output_dir / "train" / "images"
        val_img_dir = output_dir / "val" / "images"
        test_img_dir = output_dir / "test" / "images"

        train_anno_dir = output_dir / "train" / "annotations"
        val_anno_dir = output_dir / "val" / "annotations"
        test_anno_dir = output_dir / "test" / "annotations"

        for d in [train_img_dir, val_img_dir, test_img_dir,
                  train_anno_dir, val_anno_dir, test_anno_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Helper function to transfer files
        def transfer_files(names, img_dir, anno_dir):
            img_count = 0
            anno_count = 0

            for name in names:
                # Find and transfer image file
                image_files = list(dataset_dir.glob(f"{name}.*"))
                image_files = [f for f in image_files if validate_image(f)]

                if not image_files:
                    logger.warning(f"No valid image found for {name}")
                    continue

                img_src = image_files[0]
                img_dst = img_dir / img_src.name

                try:
                    if copy_files:
                        shutil.copy2(img_src, img_dst)
                    else:
                        shutil.move(img_src, img_dst)
                    img_count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer image {img_src}: {str(e)}")
                    continue

                # Find and transfer annotation file
                anno_src = annotation_dir / f"{name}.txt"
                if not anno_src.exists():
                    logger.warning(f"Annotation file not found: {anno_src}")
                    continue

                anno_dst = anno_dir / f"{name}.txt"

                try:
                    if copy_files:
                        shutil.copy2(anno_src, anno_dst)
                    else:
                        shutil.move(anno_src, anno_dst)
                    anno_count += 1
                except Exception as e:
                    logger.error(f"Failed to transfer annotation {anno_src}: {str(e)}")

            return img_count, anno_count

        # Transfer files to each split
        train_img_count, train_anno_count = transfer_files(train_names, train_img_dir, train_anno_dir)
        val_img_count, val_anno_count = transfer_files(val_names, val_img_dir, val_anno_dir)
        test_img_count, test_anno_count = transfer_files(test_names, test_img_dir, test_anno_dir)

        # Calculate category distribution
        category_counts = {}
        for cats in image_categories.values():
            for cat in cats:
                if cat not in category_counts:
                    category_counts[cat] = 0
                category_counts[cat] += 1

        # Create statistics
        stats = {
            "total_images": len(image_names),
            "train_images": train_img_count,
            "val_images": val_img_count,
            "test_images": test_img_count,
            "train_annotations": train_anno_count,
            "val_annotations": val_anno_count,
            "test_annotations": test_anno_count,
            "category_distribution": {str(k): v for k, v in category_counts.items()}
        }

        # Save split statistics
        stats_file = output_dir / "split_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Stratified split completed. Stats: {stats}")
        return stats


# models/detection/__init__.py
"""
Detection models for identifying framing members in residential construction images.
"""

from models.detection.framing_detector import FramingDetector
from models.detection.train import train_model
from models.detection.evaluate import evaluate_model
from models.detection.inference import detect_framing, visualize_detections

__all__ = [
    'FramingDetector',
    'train_model',
    'evaluate_model',
    'detect_framing',
    'visualize_detections'
]


# models/detection/model_config.py
"""
Configuration for framing member detection models.
"""

from pathlib import Path

# Framing member categories
CATEGORIES = [
    {'id': 0, 'name': 'stud'},
    {'id': 1, 'name': 'joist'},
    {'id': 2, 'name': 'rafter'},
    {'id': 3, 'name': 'beam'},
    {'id': 4, 'name': 'plate'},
    {'id': 5, 'name': 'header'},
    {'id': 6, 'name': 'blocking'},
    {'id': 7, 'name': 'electrical_box'}
]

# Color mapping for visualization
CATEGORY_COLORS = {
    'stud': (0, 255, 0),      # Green
    'joist': (255, 0, 0),     # Blue
    'rafter': (0, 0, 255),    # Red
    'beam': (0, 255, 255),     # Yellow
    'plate': (255, 0, 255),    # Magenta
    'header': (255, 255, 0),   # Cyan
    'blocking': (128, 128, 0), # Olive
    'electrical_box': (128, 0, 128) # Purple
}

# Model configuration
DEFAULT_MODEL_SIZE = 'm'  # nano, small, medium, large, extra-large
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Training configuration
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 20
DEFAULT_LR = 0.01

# Default paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / 'models' / 'weights'
CHECKPOINTS_DIR = WEIGHTS_DIR / 'checkpoints'
EXPORTS_DIR = WEIGHTS_DIR / 'exports'

# Create directories if they don't exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO-specific configuration
YOLO_CONFIG = {
    'model_type': 'yolov8',
    'pretrained_weights': f'yolov8{DEFAULT_MODEL_SIZE}.pt',
    'task': 'detect',
    'num_classes': len(CATEGORIES),
    'class_names': [cat['name'] for cat in CATEGORIES]
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'hsv_h': 0.015,    # HSV-Hue augmentation
    'hsv_s': 0.7,      # HSV-Saturation augmentation
    'hsv_v': 0.4,      # HSV-Value augmentation
    'degrees': 0.0,    # Rotation (±deg)
    'translate': 0.1,  # Translation (±fraction)
    'scale': 0.5,      # Scale (±gain)
    'shear': 0.0,      # Shear (±deg)
    'perspective': 0.0, # Perspective (±fraction), 0.0=disabled
    'flipud': 0.0,     # Flip up-down (probability)
    'fliplr': 0.5,     # Flip left-right (probability)
    'mosaic': 1.0,     # Mosaic (probability)
    'mixup': 0.0,      # Mixup (probability)
    'copy_paste': 0.0  # Copy-paste (probability)
}


# models/detection/framing_detector.py
"""
YOLOv8-based model for detecting framing members in residential construction images.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import time
from ultralytics import YOLO

from models.detection.model_config import (
    CATEGORIES, CATEGORY_COLORS, YOLO_CONFIG,
    DEFAULT_MODEL_SIZE, DEFAULT_IMG_SIZE,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    WEIGHTS_DIR, CHECKPOINTS_DIR, EXPORTS_DIR
)
from utils.logger import get_logger
from utils.exceptions import ModelInferenceError, ModelNotFoundError

logger = get_logger("framing_detector")

class FramingDetector:
    """
    A detector for framing members in residential construction images using YOLOv8.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        img_size: int = DEFAULT_IMG_SIZE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the framing detector.

        Args:
            model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
            img_size: Input image size
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            pretrained: Whether to load pretrained weights
            weights_path: Path to custom weights file
        """
        self.model_size = model_size
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Initializing FramingDetector with YOLOv8{model_size} on {self.device}")

        # Load model
        try:
            if weights_path is not None:
                # Load custom trained weights
                weights_path = Path(weights_path)
                if not weights_path.exists():
                    raise ModelNotFoundError(f"Weights file not found: {weights_path}")

                logger.info(f"Loading custom weights from {weights_path}")
                self.model = YOLO(str(weights_path))

            elif pretrained:
                # Load pretrained weights
                logger.info(f"Loading pretrained YOLOv8{model_size}")
                self.model = YOLO(f"yolov8{model_size}.pt")

                # Update number of classes if needed
                if self.model.names != YOLO_CONFIG['class_names']:
                    logger.info(f"Updating model for {len(CATEGORIES)} framing categories")
                    self.model.names = YOLO_CONFIG['class_names']
            else:
                # Initialize with random weights
                logger.info(f"Initializing YOLOv8{model_size} with random weights")
                self.model = YOLO(f"yolov8{model_size}.yaml")

            # Move model to device
            self.model.to(self.device)

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load YOLOv8 model: {str(e)}")

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        return_original: bool = False
    ) -> Dict:
        """
        Detect framing members in an image.

        Args:
            image: Image file path or numpy array
            conf_threshold: Confidence threshold for detections (overrides default)
            iou_threshold: IoU threshold for NMS (overrides default)
            return_original: Whether to include the original image in the results

        Returns:
            Dict: Detection results with keys:
                - 'detections': List of detection dictionaries
                - 'image': Original image (if return_original=True)
                - 'inference_time': Time taken for inference
        """
        # Use specified thresholds or fall back to instance defaults
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        try:
            # Track inference time
            start_time = time.time()

            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )

            inference_time = time.time() - start_time

            # Process results
            detections = []

            # Extract results from the first image (or only image)
            result = results[0]

            # Convert boxes to the desired format
            if len(result.boxes) > 0:
                # Get boxes, classes, and confidence scores
                boxes = result.boxes.xyxy.cpu().numpy() # x1, y1, x2, y2 format
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                # Format detections
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    x1, y1, x2, y2 = box

                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1

                    # Get class name
                    class_name = result.names[cls]

                    detection = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(width), float(height)],
                        'category_id': int(cls),
                        'category_name': class_name,
                        'confidence': float(score)
                    }

                    detections.append(detection)

            # Prepare return dictionary
            results_dict = {
                'detections': detections,
                'inference_time': inference_time
            }

            # Include original image if requested
            if return_original:
                if isinstance(image, (str, Path)):
                    # If image is a path, get the processed image from results
                    results_dict['image'] = result.orig_img
                else:
                    # If image is an array, use it directly
                    results_dict['image'] = image

            return results_dict

        except Exception as e:
            raise ModelInferenceError(f"Error during framing detection: {str(e)}")

    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        patience: int = 20,
        project: str = 'framing_detection',
        name: str = 'train',
        device: Optional[str] = None,
        lr0: float = 0.01,
        lrf: float = 0.01,
        save: bool = True,
        resume: bool = False,
        pretrained: bool = True,
        **kwargs
    ) -> Any:
        """
        Train the detector on a dataset.

        Args:
            data_yaml: Path to data configuration file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Input image size
            patience: Epochs to wait for no improvement before early stopping
            project: Project name for saving results
            name: Run name for this training session
            device: Device to use (None for auto-detection)
            lr0: Initial learning rate
            lrf: Final learning rate (fraction of lr0)
            save: Whether to save the model
            resume: Resume training from the last checkpoint
            pretrained: Use pretrained weights
            **kwargs: Additional arguments to pass to the trainer

        Returns:
            Training results
        """
        device = device or self.device

        logger.info(f"Training YOLOv8{self.model_size} on {device}")
        logger.info(f"Data config: {data_yaml}, Epochs: {epochs}, Batch size: {batch_size}")

        # Set up training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'project': project,
            'name': name,
            'device': device,
            'lr0': lr0,
            'lrf': lrf,
            'save': save,
            'pretrained': pretrained,
            'resume': resume
        }

        # Add any additional kwargs
        train_args.update(kwargs)

        # Start training
        try:
            results = self.model.train(**train_args)

            logger.info(f"Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def export(
        self,
        format: str = 'onnx',
        output_path: Optional[Union[str, Path]] = None,
        dynamic: bool = True,
        half: bool = True,
        simplify: bool = True
    ) -> Path:
        """
        Export the model to a deployable format.

        Args:
            format: Export format ('onnx', 'torchscript', 'openvino', etc.)
            output_path: Path to save the exported model
            dynamic: Use dynamic axes in ONNX export
            half: Export with half precision (FP16)
            simplify: Simplify the model during export

        Returns:
            Path: Path to the exported model
        """
        if output_path is None:
            # Generate default output path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = EXPORTS_DIR / f"framing_detector_{timestamp}.{format}"
        else:
            output_path = Path(output_path)

        logger.info(f"Exporting model to {format} format: {output_path}")

        try:
            # Export the model
            exported_path = self.model.export(
                format=format,
                imgsz=self.img_size,
                dynamic=dynamic,
                half=half,
                simplify=simplify
            )

            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # If the export path is different from the desired output path, move it
            if str(exported_path) != str(output_path):
                shutil.copy(exported_path, output_path)
                os.remove(exported_path)
                logger.info(f"Moved exported model to {output_path}")

            logger.info(f"Model exported successfully to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise

    def save_checkpoint(
        self,
        path: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint
            overwrite: Whether to overwrite if file exists

        Returns:
            Path: Path to the saved checkpoint
        """
        if path is None:
            # Generate default path
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = CHECKPOINTS_DIR / f"framing_detector_{timestamp}.pt"
        else:
            path = Path(path)

        # Check if file exists and overwrite is False
        if path.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint file already exists: {path}")

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model checkpoint to {path}")

        try:
            self.model.save(str(path))
            logger.info(f"Model checkpoint saved successfully")
            return path
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD
    ) -> 'FramingDetector':
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            FramingDetector: Loaded model
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ModelNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Create detector with custom weights
        detector = cls(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            pretrained=False,
            weights_path=checkpoint_path
        )

        return detector


# models/detection/train.py
"""
Training module for framing member detection models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import shutil

from models.detection.framing_detector import FramingDetector
from models.detection.model_config import (
    DEFAULT_MODEL_SIZE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_IMG_SIZE, DEFAULT_PATIENCE, DEFAULT_LR,
    WEIGHTS_DIR, AUGMENTATION_CONFIG
)
from utils.logger import get_logger

logger = get_logger("train")

def create_data_yaml(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a YAML file for YOLOv8 training.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        class_names: List of class names
        output_path: Path to save the YAML file

    Returns:
        Path: Path to the created YAML file
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    # Default output path if not specified
    if output_path is None:
        output_path = train_dir.parent / "dataset.yaml"
    else:
        output_path = Path(output_path)

    # Create the data configuration
    data_dict = {
        'path': str(train_dir.parent),
        'train': str(train_dir.relative_to(train_dir.parent)),
        'val': str(val_dir.relative_to(train_dir.parent)),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    logger.info(f"Created data YAML file: {output_path}")
    return output_path

def train_model(
    data_dir: Union[str, Path],
    model_size: str = DEFAULT_MODEL_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LR,
    pretrained: bool = True,
    augmentation: Optional[Dict] = None,
    save_checkpoint: bool = True,
    export_format: Optional[str] = 'onnx',
    project_name: str = 'framing_detection',
    run_name: Optional[str] = None
) -> Tuple[FramingDetector, Dict]:
    """
    Train a framing detection model.

    Args:
        data_dir: Directory containing the dataset with train/val subdirectories
        model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        patience: Epochs to wait for no improvement before early stopping
        learning_rate: Initial learning rate
        pretrained: Use pretrained weights
        augmentation: Augmentation parameters (None for defaults)
        save_checkpoint: Whether to save the final model checkpoint
        export_format: Format to export the model (None to skip export)
        project_name: Project name for saving results
        run_name: Run name for this training session

    Returns:
        Tuple: (Trained model, Training results)
    """
    data_dir = Path(data_dir)

    # Check if data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Expected directory structure
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Training or validation directory not found in {data_dir}")

    # Find class names from annotations
    class_names = []
    if (train_dir / "annotations").exists():
        yaml_file = list((train_dir / "annotations").glob("*.yaml"))
        if yaml_file:
            with open(yaml_file[0], 'r') as f:
                class_data = yaml.safe_load(f)
                if isinstance(class_data, dict) and 'names' in class_data:
                    class_names = list(class_data['names'].values())

    # If class names not found, use defaults from model config
    if not class_names:
        from models.detection.model_config import CATEGORIES
        class_names = [cat['name'] for cat in CATEGORIES]

    # Create data YAML file
    data_yaml = create_data_yaml(train_dir, val_dir, class_names)

    # Generate run name if not provided
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    # Initialize model
    detector = FramingDetector(model_size=model_size, pretrained=pretrained)

    # Set up augmentation parameters
    aug_params = AUGMENTATION_CONFIG.copy()
    if augmentation is not None:
        aug_params.update(augmentation)

    # Train model
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size,
        patience=patience,
        project=project_name,
        name=run_name,
        lr0=learning_rate,
        augment=True,
        **aug_params
    )

    # Save checkpoint if requested
    if save_checkpoint:
        checkpoint_path = WEIGHTS_DIR / f"{project_name}_{run_name}.pt"
        detector.save_checkpoint(checkpoint_path)

    # Export model if requested
    if export_format:
        export_path = WEIGHTS_DIR / f"{project_name}_{run_name}.{export_format}"
        detector.export(format=export_format, output_path=export_path)

    return detector, results


# models/detection/evaluate.py
"""
Evaluation functions for framing member detection models.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.detection.framing_detector import FramingDetector
from data.utils.image_utils import list_images
from utils.logger import get_logger

logger = get_logger("evaluate")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box in format [x, y, width, height]
        box2: Second box in format [x, y, width, height]

    Returns:
        float: IoU value
    """
    # Convert from [x, y, width, height] to [x1, y1, x2, y2]
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]

    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_precision_recall(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> Tuple[Dict, Dict]:
    """
    Calculate precision and recall for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for a true positive
        score_threshold: Minimum confidence score to consider

    Returns:
        Tuple: (Precision per class, Recall per class)
    """
    # Filter predictions by score threshold
    filtered_preds = [p for p in predictions if p['confidence'] >= score_threshold]

    # Group by category
    pred_by_class = {}
    gt_by_class = {}

    for pred in filtered_preds:
        cat_id = pred['category_id']
        if cat_id not in pred_by_class:
            pred_by_class[cat_id] = []
        pred_by_class[cat_id].append(pred)

    for gt in ground_truth:
        cat_id = gt['category_id']
        if cat_id not in gt_by_class:
            gt_by_class[cat_id] = []
        gt_by_class[cat_id].append(gt)

    # Calculate precision and recall per class
    precision = {}
    recall = {}

    for cat_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds = pred_by_class.get(cat_id, [])
        gts = gt_by_class.get(cat_id, [])

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        # Initialize tracking variables
        tp = 0  # True positives
        fp = 0  # False positives

        # Track which ground truths have been matched
        gt_matched = [False] * len(gts)

        for pred in preds:
            pred_bbox = pred['bbox']
            matched = False

            # Check if prediction matches any unmatched ground truth
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                gt_bbox = gt['bbox']
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold:
                    tp += 1
                    gt_matched[gt_idx] = True
                    matched = True
                    break

            if not matched:
                fp += 1

        # Calculate metrics
        if tp + fp > 0:
            precision[cat_id] = tp / (tp + fp)
        else:
            precision[cat_id] = 0.0

        if len(gts) > 0:
            recall[cat_id] = tp / len(gts)
        else:
            recall[cat_id] = 0.0

    return precision, recall

def calculate_average_precision(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    num_points: int = 11
) -> Dict:
    """
    Calculate Average Precision (AP) for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for a true positive
        num_points: Number of points to sample for the PR curve

    Returns:
        Dict: AP per class
    """
    # Group predictions by category
    pred_by_class = {}
    gt_by_class = {}

    for pred in predictions:
        cat_id = pred['category_id']
        if cat_id not in pred_by_class:
            pred_by_class[cat_id] = []
        pred_by_class[cat_id].append(pred)

    for gt in ground_truth:
        cat_id = gt['category_id']
        if cat_id not in gt_by_class:
            gt_by_class[cat_id] = []
        gt_by_class[cat_id].append(gt)

    # Calculate AP per class
    ap_per_class = {}

    for cat_id in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds = pred_by_class.get(cat_id, [])
        gts = gt_by_class.get(cat_id, [])

        if not gts:
            ap_per_class[cat_id] = 0.0
            continue

        if not preds:
            ap_per_class[cat_id] = 0.0
            continue

        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision and recall at each detection
        tp = 0  # True positives
        fp = 0  # False positives

        # Track which ground truths have been matched
        gt_matched = [False] * len(gts)

        precisions = []
        recalls = []

        for pred_idx, pred in enumerate(preds):
            pred_bbox = pred['bbox']
            matched = False

            # Check if prediction matches any unmatched ground truth
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                gt_bbox = gt['bbox']
                iou = calculate_iou(pred_bbox, gt_bbox)

                if iou >= iou_threshold:
                    tp += 1
                    gt_matched[gt_idx] = True
                    matched = True
                    break

            if not matched:
                fp += 1

            # Calculate current precision and recall
            precision = tp / (tp + fp)
            recall = tp / len(gts)

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP by interpolating the precision-recall curve
        ap = 0.0

        # Use standard 11-point interpolation
        for t in np.linspace(0.0, 1.0, num_points):
            # Find indices where recall >= t
            indices = [i for i, r in enumerate(recalls) if r >= t]

            if indices:
                p_max = max([precisions[i] for i in indices])
                ap += p_max / num_points

        ap_per_class[cat_id] = ap

    return ap_per_class

def evaluate_model(
    model: FramingDetector,
    test_data_dir: Union[str, Path],
    annotation_dir: Optional[Union[str, Path]] = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    output_file: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Evaluate a framing detector model on test data.

    Args:
        model: The detector model to evaluate
        test_data_dir: Directory containing test images
        annotation_dir: Directory containing ground truth annotations
        iou_threshold: IoU threshold for a true positive
        conf_threshold: Confidence threshold for detections
        output_file: Path to save evaluation results

    Returns:
        Dict: Evaluation metrics
    """
    test_data_dir = Path(test_data_dir)

    # Find test images
    if (test_data_dir / "images").exists():
        test_data_dir = test_data_dir / "images"

    # Find all images
    image_paths = list_images(test_data_dir)

    if not image_paths:
        raise ValueError(f"No valid images found in {test_data_dir}")

    logger.info(f"Evaluating model on {len(image_paths)} test images")

    # Load annotations if provided
    ground_truth = []

    if annotation_dir is not None:
        annotation_dir = Path(annotation_dir)

        # Check if it's COCO format (JSON)
        coco_files = list(annotation_dir.glob("*.json"))
        if coco_files:
            # Load COCO annotations
            with open(coco_files[0], 'r') as f:
                coco_data = json.load(f)

            # Extract annotations
            for anno in coco_data.get("annotations", []):
                gt = {
                    'image_id': anno['image_id'],
                    'category_id': anno['category_id'],
                    'bbox': anno['bbox'],
                    'area': anno['area']
                }
                ground_truth.append(gt)

            # Create mapping from filename to image_id
            filename_to_id = {}
            for img in coco_data.get("images", []):
                filename_to_id[img['file_name']] = img['id']

        else:
            # Assume YOLO format (one .txt file per image)
            logger.info("No COCO annotations found, looking for YOLO format")

            # Load class names if available
            class_names = []
            yaml_files = list(annotation_dir.glob("*.yaml"))
            if yaml_files:
                with open(yaml_files[0], 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if isinstance(yaml_data, dict) and 'names' in yaml_data:
                        class_names = list(yaml_data['names'].values())

            if not class_names:
                from models.detection.model_config import CATEGORIES
                class_names = [cat['name'] for cat in CATEGORIES]

            # Process each image and its corresponding annotation
            from data.annotation.annotation_converter import AnnotationConverter
            converter = AnnotationConverter()

            for img_path in image_paths:
                img_stem = img_path.stem
                anno_path = annotation_dir / f"{img_stem}.txt"

                if not anno_path.exists():
                    logger.warning(f"No annotation found for {img_path}")
                    continue

                # Load image to get dimensions
                import cv2
                img = cv2.imread(str(img_path))
                height, width = img.shape[:2]

                # Read YOLO annotations
                with open(anno_path, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        if not line.strip():
                            continue

                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id, x_center, y_center, w, h = map(float, parts)

                        # Convert YOLO to absolute coordinates
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w = w * width
                        h = h * height

                        gt = {
                            'image_id': img_stem,
                            'category_id': int(class_id),
                            'bbox': [x, y, w, h],
                            'area': w * h
                        }
                        ground_truth.append(gt)

    # Run inference on all test images
    all_predictions = []
    total_inference_time = 0

    for img_path in tqdm(image_paths, desc="Evaluating"):
        try:
            # Run detection
            result = model.detect(
                image=str(img_path),
                conf_threshold=conf_threshold
            )

            # Get image ID (either from filename to ID mapping or use filename)
            if annotation_dir is not None and 'filename_to_id' in locals():
                img_id = filename_to_id.get(img_path.name, img_path.stem)
            else:
                img_id = img_path.stem

            # Add image ID to each detection
            for det in result['detections']:
                det['image_id'] = img_id
                all_predictions.append(det)

            total_inference_time += result['inference_time']

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

    # Calculate metrics
    metrics = {
        'num_images': len(image_paths),
        'num_predictions': len(all_predictions),
        'num_ground_truth': len(ground_truth),
        'average_inference_time': total_inference_time / len(image_paths) if image_paths else 0,
        'total_inference_time': total_inference_time
    }

    # Calculate precision and recall if ground truth is available
    if ground_truth:
        precision, recall = calculate_precision_recall(
            all_predictions, ground_truth, iou_threshold, conf_threshold
        )

        ap = calculate_average_precision(
            all_predictions, ground_truth, iou_threshold
        )

        # Calculate mAP
        if ap:
            mAP = sum(ap.values()) / len(ap)
        else:
            mAP = 0.0

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['AP'] = ap
        metrics['mAP'] = mAP

    # Save results if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation results saved to {output_file}")

    return metrics

def plot_precision_recall_curve(
    precision: Dict[int, float],
    recall: Dict[int, float],
    class_names: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot precision-recall curves.

    Args:
        precision: Precision values per class
        recall: Recall values per class
        class_names: List of class names
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Get class names if not provided
    if class_names is None:
        from models.detection.model_config import CATEGORIES
        class_names = [cat['name'] for cat in CATEGORIES]

    # Plot each class
    for class_id in precision.keys():
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"

        plt.plot(recall[class_id], precision[class_id], label=f"{class_name} (AP={precision[class_id]:.2f})")

    # Add mean average precision
    map_value = sum(precision.values()) / len(precision) if precision else 0
    plt.title(f"Precision-Recall Curve (mAP = {map_value:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best")
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
        logger.info(f"Precision-recall curve saved to {output_file}")
    else:
        plt.show()


# models/detection/inference.py
"""
Inference utilities for framing member detection models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import time

from models.detection.framing_detector import FramingDetector
from models.detection.model_config import CATEGORY_COLORS
from utils.logger import get_logger

logger = get_logger("inference")

def detect_framing(
    detector: FramingDetector,
    image_path: Union[str, Path, np.ndarray],
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None
) -> Dict:
    """
    Detect framing members in an image.

    Args:
        detector: The framing detector model
        image_path: Path to image or image array
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        Dict: Detection results
    """
    logger.debug(f"Detecting framing in image: {image_path if isinstance(image_path, (str, Path)) else 'array'}")

    return detector.detect(
        image=image_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        return_original=True
    )

def visualize_detections(
    detection_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize detection results on an image.

    Args:
        detection_result: Detection results from detect_framing
        output_path: Path to save the visualization
        show_confidence: Whether to show confidence scores
        line_thickness: Thickness of bounding box lines
        font_scale: Size of font for labels

    Returns:
        np.ndarray: Image with visualized detections
    """
    # Make sure image is included in results
    if 'image' not in detection_result:
        raise ValueError("Detection result must include the 'image' key")

    # Get image and detections
    image = detection_result['image'].copy()
    detections = detection_result['detections']

    # Draw each detection
    for det in detections:
        # Get bounding box
        x, y, w, h = [int(v) for v in det['bbox']]

        # Get category and color
        category_name = det['category_name']
        color = CATEGORY_COLORS.get(category_name, (0, 255, 0))  # Default to green

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, line_thickness)

        # Prepare label text
        if show_confidence:
            label = f"{category_name}: {det['confidence']:.2f}"
        else:
            label = category_name

        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
        cv2.rectangle(image, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)

        # Draw label
        cv2.putText(
            image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), line_thickness // 2
        )

    # Save image if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Visualization saved to {output_path}")

    return image

def batch_inference(
    detector: FramingDetector,
    image_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    conf_threshold: Optional[float] = None,
    visualize: bool = True
) -> List[Dict]:
    """
    Run inference on a batch of images.

    Args:
        detector: The framing detector model
        image_paths: List of paths to images
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold for detections
        visualize: Whether to create and save visualizations

    Returns:
        List[Dict]: Detection results for each image
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for img_path in image_paths:
        img_path = Path(img_path)

        try:
            # Run detection
            result = detect_framing(
                detector=detector,
                image_path=img_path,
                conf_threshold=conf_threshold
            )

            # Save result
            results.append(result)

            # Create visualization if requested
            if visualize and output_dir is not None:
                vis_path = output_dir / f"vis_{img_path.name}"
                visualize_detections(result, output_path=vis_path)

        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")

    return results

def analyze_framing_members(
    detection_result: Dict,
    spacing_tolerance: float = 0.1
) -> Dict:
    """
    Analyze framing members to extract structural information.

    Args:
        detection_result: Detection results from detect_framing
        spacing_tolerance: Tolerance for spacing consistency (as fraction)

    Returns:
        Dict: Analysis results
    """
    detections = detection_result['detections']

    # Separate detections by category
    by_category = {}
    for det in detections:
        cat_name = det['category_name']
        if cat_name not in by_category:
            by_category[cat_name] = []
        by_category[cat_name].append(det)

    # Analyze each category
    analysis = {
        'member_counts': {cat: len(dets) for cat, dets in by_category.items()},
        'spacing_analysis': {},
        'orientation': {}
    }

    # Function to calculate spacing between parallel members
    def analyze_spacing(members, horizontal=True):
        # Sort members by position
        if horizontal:
            # For horizontal members (like joists), sort by y-coordinate
            sorted_members = sorted(members, key=lambda x: x['bbox'][1])
        else:
            # For vertical members (like studs), sort by x-coordinate
            sorted_members = sorted(members, key=lambda x: x['bbox'][0])

        if len(sorted_members) < 2:
            return None

        # Calculate spacing between adjacent members
        spacings = []
        for i in range(len(sorted_members) - 1):
            if horizontal:
                # For horizontal members, measure center-to-center y distance
                y1 = sorted_members[i]['bbox'][1] + sorted_members[i]['bbox'][3] / 2
                y2 = sorted_members[i+1]['bbox'][1] + sorted_members[i+1]['bbox'][3] / 2
                spacing = abs(y2 - y1)
            else:
                # For vertical members, measure center-to-center x distance
                x1 = sorted_members[i]['bbox'][0] + sorted_members[i]['bbox'][2] / 2
                x2 = sorted_members[i+1]['bbox'][0] + sorted_members[i+1]['bbox'][2] / 2
                spacing = abs(x2 - x1)

            spacings.append(spacing)

        if not spacings:
            return None

        # Calculate statistics
        mean_spacing = sum(spacings) / len(spacings)
        min_spacing = min(spacings)
        max_spacing = max(spacings)

        # Check consistency
        variation = (max_spacing - min_spacing) / mean_spacing
        is_consistent = variation <= spacing_tolerance

        return {
            'mean_spacing': mean_spacing,
            'min_spacing': min_spacing,
            'max_spacing': max_spacing,
            'is_consistent': is_consistent,
            'variation': variation
        }

    # Analyze studs (vertical members)
    if 'stud' in by_category:
        analysis['spacing_analysis']['stud'] = analyze_spacing(by_category['stud'], horizontal=False)

    # Determine orientation
    orientations = []
    for stud in by_category['stud']:
        w, h = stud['bbox'][2], stud['bbox'][3]
        orientations.append('vertical' if h > w else 'horizontal')

    analysis['orientation']['stud'] = {
        'vertical': orientations.count('vertical'),
        'horizontal': orientations.count('horizontal')
    }

    # Analyze joists (horizontal members)
    if 'joist' in by_category:
        analysis['spacing_analysis']['joist'] = analyze_spacing(by_category['joist'], horizontal=True)

    # Determine orientation
    orientations = []
    for joist in by_category['joist']:
        w, h = joist['bbox'][2], joist['bbox'][3]
        orientations.append('horizontal' if w > h else 'vertical')

    analysis['orientation']['joist'] = {
        'horizontal': orientations.count('horizontal'),
        'vertical': orientations.count('vertical')
    }

    # Count other members
    for cat in by_category:
        if cat not in ('stud', 'joist'):
            analysis['member_counts'][cat] = len(by_category[cat])

    return analysis


# models/measurements/__init__.py
"""
Measurement estimation module for the electrician time estimation application.
This module provides tools for calculating distances, dimensions, and wiring paths
from detected framing members in residential construction images.
"""

from models.measurements.measurement_estimator import MeasurementEstimator
from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from models.measurements.visualization import (
    visualize_measurements,
    visualize_wiring_path,
    visualize_scale_calibration
)

__all__ = [
    'MeasurementEstimator',
    'ReferenceScale',
    'ScaleCalibration',
    'SpacingCalculator',
    'DimensionEstimator',
    'PathCalculator',
    'visualize_measurements',
    'visualize_wiring_path',
    'visualize_scale_calibration'
]


# models/measurements/measurement_estimator.py
"""
Main measurement estimation class for analyzing framing members.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import json

from models.measurements.reference_scale import ReferenceScale, ScaleCalibration
from models.measurements.spacing_calculator import SpacingCalculator
from models.measurements.dimension_estimator import DimensionEstimator
from models.measurements.path_calculator import PathCalculator
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("measurement_estimator")

class MeasurementEstimator:
    """
    Class for estimating measurements from detected framing members.
    """

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None,
        confidence_threshold: float = 0.7,
        detection_threshold: float = 0.25
    ):
        """
        Initialize the measurement estimator.

        Args:
            pixels_per_inch: Calibration value (pixels per inch)
            calibration_data: Pre-computed calibration data
            confidence_threshold: Threshold for including detections in measurements
            detection_threshold: Threshold for detection confidence
        """
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold

        # Initialize the reference scale
        self.reference_scale = ReferenceScale(
            pixels_per_inch=pixels_per_inch,
            calibration_data=calibration_data
        )

        # Initialize measurement components
        self.spacing_calculator = SpacingCalculator(self.reference_scale)
        self.dimension_estimator = DimensionEstimator(self.reference_scale)
        self.path_calculator = PathCalculator(self.reference_scale)

        # Store measurement history
        self.last_measurement_result = None

        logger.info("Measurement estimator initialized")

    def calibrate_from_reference(
        self,
        image: np.ndarray,
        reference_points: List[Tuple[int, int]],
        reference_distance: float,
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using reference points.

        Args:
            image: Input image array
            reference_points: List of two (x, y) points defining a known distance
            reference_distance: Known distance between points
            units: Units of the reference distance ("inches", "feet", "mm", "cm", "m")

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_points(
                reference_points, reference_distance, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def calibrate_from_known_object(
        self,
        image: np.ndarray,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        units: str = "inches"
    ) -> Dict:
        """
        Calibrate the measurement system using a known object.

        Args:
            image: Input image array
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            units: Units of the reference dimensions

        Returns:
            Dict: Calibration result
        """
        try:
            calibration = self.reference_scale.calibrate_from_object(
                object_bbox, object_dimensions, units
            )

            logger.info(f"Scale calibrated: {calibration['pixels_per_unit']} pixels per {units}")
            return calibration

        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def analyze_framing_measurements(
        self,
        detection_result: Dict,
        calibration_check: bool = True
    ) -> Dict:
        """
        Analyze framing member detections to extract measurements.

        Args:
            detection_result: Detection results from framing detector
            calibration_check: Whether to verify calibration first

        Returns:
            Dict: Measurement analysis results
        """
        if calibration_check and not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Filter detections based on confidence
            detections = [det for det in detection_result['detections']
                          if det['confidence'] >= self.detection_threshold]

            if not detections:
                logger.warning("No valid detections found for measurement analysis")
                return {
                    "status": "warning",
                    "message": "No valid detections found",
                    "measurements": {}
                }

            # Extract image if available
            image = detection_result.get('image')

            # Calculate spacing measurements
            spacing_results = self.spacing_calculator.calculate_spacings(detections, image)

            # Estimate framing dimensions
            dimension_results = self.dimension_estimator.estimate_dimensions(detections, image)

            # Collect all measurements and calculate overall confidence
            measurements = {
                "spacing": spacing_results,
                "dimensions": dimension_results,
                "unit": self.reference_scale.get_unit(),
                "pixels_per_unit": self.reference_scale.get_pixels_per_unit()
            }

            # Calculate overall confidence score
            detection_confs = [det['confidence'] for det in detections]
            avg_detection_conf = sum(detection_confs) / len(detections) if detections else 0

            spacing_conf = spacing_results.get("confidence", 0) if spacing_results else 0
            dimension_conf = dimension_results.get("confidence", 0) if dimension_results else 0
            scale_conf = self.reference_scale.get_calibration_confidence()

            overall_confidence = 0.4 * avg_detection_conf + 0.3 * spacing_conf + \
                                 0.2 * dimension_conf + 0.1 * scale_conf

            # Store the results for later reference
            self.last_measurement_result = {
                "status": "success",
                "message": "Measurement analysis completed",
                "measurements": measurements,
                "confidence": overall_confidence
            }

            return self.last_measurement_result

        except Exception as e:
            error_msg = f"Measurement analysis failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def estimate_wiring_path(
        self,
        detection_result: Dict,
        path_points: List[Tuple[int, int]],
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Estimate the total distance of a wiring path.

        Args:
            detection_result: Detection results from framing detector
            path_points: List of (x, y) points defining the wiring path
            drill_points: List of (x, y) points where drilling is required

        Returns:
            Dict: Wiring path analysis
        """
        if not self.reference_scale.is_calibrated():
            error_msg = "Reference scale is not calibrated. Call calibrate_* methods first."
            logger.error(error_msg)
            raise MeasurementError(error_msg)

        try:
            # Extract image if available
            image = detection_result.get('image')

            # Calculate path measurements
            path_results = self.path_calculator.calculate_path(
                path_points, image, drill_points=drill_points
            )

            # Calculate drill points if not provided
            if drill_points is None and 'detections' in detection_result:
                detected_drill_points = self.path_calculator.identify_drill_points(
                    path_points, detection_result['detections'], image
                )
                path_results['detected_drill_points'] = detected_drill_points

            # Store the results
            self.last_path_result = path_results

            return path_results

        except Exception as e:
            error_msg = f"Wiring path estimation failed: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)

    def save_measurements(self, output_file: Union[str, Path]) -> None:
        """
        Save the last measurement results to a file.

        Args:
            output_file: Path to save the measurement data
        """
        if self.last_measurement_result is None:
            logger.warning("No measurements to save")
            return

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w') as f:
                json.dump(self.last_measurement_result, f, indent=2)

            logger.info(f"Measurement results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save measurements: {str(e)}")

    def load_measurements(self, input_file: Union[str, Path]) -> Dict:
        """
        Load measurement results from a file.

        Args:
            input_file: Path to the measurement data file

        Returns:
            Dict: Loaded measurement data
        """
        input_file = Path(input_file)

        if not input_file.exists():
            error_msg = f"Measurement file not found: {input_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(input_file, 'r') as f:
                measurement_data = json.load(f)

            self.last_measurement_result = measurement_data

            # Update calibration if present
            if 'measurements' in measurement_data and 'pixels_per_unit' in measurement_data['measurements']:
                unit = measurement_data['measurements'].get('unit', 'inches')
                pixels_per_unit = measurement_data['measurements']['pixels_per_unit']

                self.reference_scale.set_calibration(pixels_per_unit, unit)

            logger.info(f"Measurement results loaded from {input_file}")
            return measurement_data

        except Exception as e:
            error_msg = f"Failed to load measurements: {str(e)}"
            logger.error(error_msg)
            raise MeasurementError(error_msg)


# models/measurements/reference_scale.py
"""
Reference scale handling for measurement calibration.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass

from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("reference_scale")

@dataclass
class ScaleCalibration:
    """Scale calibration data structure."""
    pixels_per_unit: float
    unit: str
    confidence: float
    method: str
    reference_data: Dict


class ReferenceScale:
    """
    Class for handling reference scale calibration and conversion.
    """

    # Standard conversion factors to inches
    UNIT_TO_INCHES = {
        "inches": 1.0,
        "feet": 12.0,
        "mm": 0.0393701,
        "cm": 0.393701,
        "m": 39.3701
    }

    def __init__(
        self,
        pixels_per_inch: Optional[float] = None,
        calibration_data: Optional[Dict] = None
    ):
        """
        Initialize the reference scale.

        Args:
            pixels_per_inch: Initial calibration value
            calibration_data: Pre-computed calibration data
        """
        self.calibration = None

        # Initialize from pixels_per_inch if provided
        if pixels_per_inch is not None:
            self.calibration = ScaleCalibration(
                pixels_per_unit=pixels_per_inch,
                unit="inches",
                confidence=0.9,  # High confidence since directly provided
                method="manual",
                reference_data={"pixels_per_inch": pixels_per_inch}
            )

        # Or from calibration data if provided
        elif calibration_data is not None:
            self.calibration = ScaleCalibration(
                pixels_per_unit=calibration_data.get("pixels_per_unit", 0),
                unit=calibration_data.get("unit", "inches"),
                confidence=calibration_data.get("confidence", 0.5),
                method=calibration_data.get("method", "loaded"),
                reference_data=calibration_data.get("reference_data", {})
            )

    def is_calibrated(self) -> bool:
        """
        Check if the scale is calibrated.

        Returns:
            bool: True if calibrated
        """
        return self.calibration is not None and self.calibration.pixels_per_unit > 0

    def get_pixels_per_unit(self) -> float:
        """
        Get the current pixels per unit value.

        Returns:
            float: Pixels per unit
        """
        if not self.is_calibrated():
            return 0.0
        return self.calibration.pixels_per_unit

    def get_unit(self) -> str:
        """
        Get the current unit.

        Returns:
            str: Current unit
        """
        if not self.is_calibrated():
            return "uncalibrated"
        return self.calibration.unit

    def get_calibration_confidence(self) -> float:
        """
        Get the confidence level of the current calibration.

        Returns:
            float: Confidence level (0-1)
        """
        if not self.is_calibrated():
            return 0.0
        return self.calibration.confidence

    def get_calibration_data(self) -> Dict:
        """
        Get the full calibration data.

        Returns:
            Dict: Calibration data
        """
        if not self.is_calibrated():
            return {
                "status": "uncalibrated",
                "pixels_per_unit": 0.0,
                "unit": "uncalibrated",
                "confidence": 0.0
            }

        return {
            "status": "calibrated",
            "pixels_per_unit": self.calibration.pixels_per_unit,
            "unit": self.calibration.unit,
            "confidence": self.calibration.confidence,
            "method": self.calibration.method,
            "reference_data": self.calibration.reference_data
        }

    def set_calibration(self, pixels_per_unit: float, unit: str = "inches") -> None:
        """
        Set calibration manually.

        Args:
            pixels_per_unit: Pixels per unit value
            unit: Unit of measurement
        """
        if pixels_per_unit <= 0:
            raise ValueError("Pixels per unit must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=0.9,  # High confidence for manual setting
            method="manual",
            reference_data={"direct_setting": True}
        )

        logger.info(f"Manual calibration set: {pixels_per_unit} pixels per {unit}")

    def calibrate_from_points(
        self,
        points: List[Tuple[int, int]],
        known_distance: float,
        unit: str = "inches"
    ) -> Dict:
        """
        Calibrate from two points with a known distance.

        Args:
            points: List of two (x, y) points
            known_distance: Known distance between points
            unit: Unit of the known distance

        Returns:
            Dict: Calibration result
        """
        if len(points) != 2:
            raise ValueError("Exactly two points required for calibration")

        if known_distance <= 0:
            raise ValueError("Known distance must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        # Calculate pixel distance
        x1, y1 = points[0]
        x2, y2 = points[1]
        pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if pixel_distance <= 0:
            raise MeasurementError("Points are too close for accurate calibration")

        # Calculate pixels per unit
        pixels_per_unit = pixel_distance / known_distance

        # Set calibration
        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=0.95,  # Higher confidence for direct measurement
            method="point_distance",
            reference_data={
                "points": points,
                "known_distance": known_distance,
                "pixel_distance": pixel_distance
            }
        )

        logger.info(f"Calibrated from points: {pixels_per_unit} pixels per {unit}")

        return {
            "status": "calibrated",
            "pixels_per_unit": pixels_per_unit,
            "unit": unit,
            "confidence": 0.95,
            "method": "point_distance"
        }

    def calibrate_from_object(
        self,
        object_bbox: Tuple[int, int, int, int],
        object_dimensions: Tuple[float, float],
        unit: str = "inches"
    ) -> Dict:
        """
        Calibrate from an object with known dimensions.

        Args:
            object_bbox: Bounding box of reference object [x, y, width, height]
            object_dimensions: Known real-world dimensions [width, height]
            unit: Unit of the known dimensions

        Returns:
            Dict: Calibration result
        """
        if len(object_bbox) != 4 or len(object_dimensions) != 2:
            raise ValueError("Invalid bounding box or dimensions format")

        if min(object_dimensions) <= 0:
            raise ValueError("Object dimensions must be positive")

        if unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit: {unit}")

        # Extract dimensions
        bbox_width, bbox_height = object_bbox[2], object_bbox[3]
        real_width, real_height = object_dimensions

        # Calculate pixels per unit for width and height
        pixels_per_unit_width = bbox_width / real_width
        pixels_per_unit_height = bbox_height / real_height

        # Average the two measurements, but weigh by the larger dimension
        # for better accuracy
        total_real = real_width + real_height
        pixels_per_unit = (
            (pixels_per_unit_width * real_width / total_real) +
            (pixels_per_unit_height * real_height / total_real)
        )

        # Calculate confidence based on aspect ratio consistency
        real_aspect = real_width / real_height if real_height != 0 else 1
        bbox_aspect = bbox_width / bbox_height if bbox_height != 0 else 1

        aspect_diff = abs(real_aspect - bbox_aspect) / max(real_aspect, bbox_aspect)
        aspect_confidence = max(0, 1 - aspect_diff)

        # Adjust confidence by object size (larger objects generally allow more accurate calibration)
        size_factor = min(1.0, max(bbox_width, bbox_height) / 300)  # Normalize to 0-1

        confidence = 0.85 * aspect_confidence + 0.15 * size_factor

        # Set calibration
        self.calibration = ScaleCalibration(
            pixels_per_unit=pixels_per_unit,
            unit=unit,
            confidence=confidence,
            method="object_dimensions",
            reference_data={
                "object_bbox": object_bbox,
                "object_dimensions": object_dimensions,
                "pixels_per_unit_width": pixels_per_unit_width,
                "pixels_per_unit_height": pixels_per_unit_height
            }
        )

        logger.info(f"Calibrated from object: {pixels_per_unit} pixels per {unit} (confidence: {confidence:.2f})")

        return {
            "status": "calibrated",
            "pixels_per_unit": pixels_per_unit,
            "unit": unit,
            "confidence": confidence,
            "method": "object_dimensions"
        }

    def pixels_to_real_distance(self, pixels: float) -> float:
        """
        Convert pixel distance to real-world distance.

        Args:
            pixels: Distance in pixels

        Returns:
            float: Real-world distance in the calibrated unit
        """
        if not self.is_calibrated():
            raise MeasurementError("Cannot convert distance: Scale not calibrated")

        return pixels / self.calibration.pixels_per_unit

    def real_distance_to_pixels(self, distance: float) -> float:
        """
        Convert real-world distance to pixels.

        Args:
            distance: Real-world distance in the calibrated unit

        Returns:
            float: Distance in pixels
        """
        if not self.is_calibrated():
            raise MeasurementError("Cannot convert distance: Scale not calibrated")

        return distance * self.calibration.pixels_per_unit

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert a measurement between different units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            float: Converted value
        """
        if from_unit not in self.UNIT_TO_INCHES or to_unit not in self.UNIT_TO_INCHES:
            raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

        # Convert to inches first, then to target unit
        inches = value * self.UNIT_TO_INCHES[from_unit]
        return inches / self.UNIT_TO_INCHES[to_unit]


# models/measurements/spacing_calculator.py
"""
Module for calculating spacings between framing members.
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("spacing_calculator")

class SpacingCalculator:
    """
    Class for calculating spacings between framing members.
    """

    def __init__(self, reference_scale: ReferenceScale):
        """
        Initialize the spacing calculator.

        Args:
            reference_scale: Reference scale for conversions
        """
        self.reference_scale = reference_scale

    def calculate_spacings(
        self,
        detections: List[Dict],
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate spacings between framing members.

        Args:
            detections: List of detection dictionaries
            image: Original image (optional, for visualization)

        Returns:
            Dict: Spacing analysis results
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        if not detections:
            return {
                "status": "error",
                "message": "No detections to analyze",
                "spacings": {}
            }

        # Group detections by category
        categories = defaultdict(list)
        for det in detections:
            cat_name = det['category_name']
            categories[cat_name].append(det)

        # Calculate spacings for each category
        results = {}
        confidence_scores = []

        # Process each category with multiple members
        for cat_name, members in categories.items():
            if len(members) < 2:
                continue

            # Get center points and analyze member orientation
            centers = []
            widths = []
            heights = []

            for member in members:
                bbox = member['bbox']
                x, y, w, h = bbox
                center_x = x + w / 2
                center_y = y + h / 2
                centers.append((center_x, center_y))
                widths.append(w)
                heights.append(h)

            # Determine if members are predominantly horizontal or vertical
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)

            if avg_height > avg_width * 1.5:
                orientation = "vertical"  # Like wall studs
            elif avg_width > avg_height * 1.5:
                orientation = "horizontal" # Like floor joists
            else:
                orientation = "mixed"

            # Sort centers based on orientation
            if orientation == "vertical":
                # Sort by x-coordinate for side-by-side spacing
                centers.sort(key=lambda p: p[0])

                # Calculate spacings between adjacent centers
                spacings_px = [centers[i+1][0] - centers[i][0] for i in range(len(centers)-1)]

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # Calculate center-to-center and clear spacings
                avg_width_px = avg_width
                avg_width_real = self.reference_scale.pixels_to_real_distance(avg_width_px)

                clear_spacings_real = [max(0, s - avg_width_real) for s in spacings_real]

            elif orientation == "horizontal":
                # Sort by y-coordinate for top-to-bottom spacing
                centers.sort(key=lambda p: p[1])

                # Calculate spacings between adjacent centers
                spacings_px = [centers[i+1][1] - centers[i][1] for i in range(len(centers)-1)]

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # Calculate center-to-center and clear spacings
                avg_height_px = avg_height
                avg_height_real = self.reference_scale.pixels_to_real_distance(avg_height_px)

                clear_spacings_real = [max(0, s - avg_height_real) for s in spacings_real]

            else:
                # For mixed orientation, calculate pairwise distances
                spacings_px = []
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dist = math.sqrt((centers[j][0] - centers[i][0])**2 +
                                         (centers[j][1] - centers[i][1])**2)
                        spacings_px.append(dist)

                # Convert to real distances
                spacings_real = [self.reference_scale.pixels_to_real_distance(s) for s in spacings_px]

                # No clear spacing calculation for mixed orientation
                clear_spacings_real = None

            # Calculate statistics
            if spacings_real:
                mean_spacing = sum(spacings_real) / len(spacings_real)
                min_spacing = min(spacings_real)
                max_spacing = max(spacings_real)

                # Standard deviation for variability assessment
                if len(spacings_real) > 1:
                    std_dev = math.sqrt(sum((s - mean_spacing)**2 for s in spacings_real) / len(spacings_real))
                    cv = std_dev / mean_spacing if mean_spacing > 0 else float('inf') # Coefficient of variation
                else:
                    std_dev = 0
                    cv = 0

                # Determine if spacing is on standard centers
                # Convert to inches for standard comparison
                if self.reference_scale.get_unit() != "inches":
                    spacings_inches = [
                        self.reference_scale.convert_units(s, self.reference_scale.get_unit(), "inches")
                        for s in spacings_real
                    ]
                    mean_inches = sum(spacings_inches) / len(spacings_inches)
                else:
                    mean_inches = mean_spacing

                # Check if close to common framing spacings
                common_spacings = [16, 24, 12, 8, 19.2]  # Common framing centers in inches
                closest_standard = min(common_spacings, key=lambda x: abs(x - mean_inches))
                distance_to_standard = abs(closest_standard - mean_inches)

                is_standard = distance_to_standard < 1.5 # Within 1.5 inches

                # Calculate confidence based on variability and standard matching
                consistency_conf = max(0, 1 - min(cv * 2, 1)) # Lower CV = higher confidence
                standard_conf = 1.0 if is_standard else max(0, 1 - distance_to_standard / 8)

                spacing_confidence = 0.7 * consistency_conf + 0.3 * standard_conf
                confidence_scores.append(spacing_confidence)

                # Store results for this category
                results[cat_name] = {
                    "orientation": orientation,
                    "mean_spacing": mean_spacing,
                    "min_spacing": min_spacing,
                    "max_spacing": max_spacing,
                    "standard_deviation": std_dev,
                    "coefficient_of_variation": cv,
                    "closest_standard": closest_standard if self.reference_scale.get_unit() == "inches" else
                                        self.reference_scale.convert_units(closest_standard, "inches",
                                                                         self.reference_scale.get_unit()),
                    "is_standard": is_standard,
                    "unit": self.reference_scale.get_unit(),
                    "confidence": spacing_confidence,
                    "center_points": centers,
                    "spacings": spacings_real
                }

                # Add clear spacings if available
                if clear_spacings_real is not None:
                    results[cat_name]["clear_spacings"] = clear_spacings_real
                    results[cat_name]["mean_clear_spacing"] = sum(clear_spacings_real) / len(clear_spacings_real)

        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            "status": "success",
            "spacings": results,
            "confidence": overall_confidence
        }

    def find_standard_spacing(
        self,
        detections: List[Dict],
        category_name: Optional[str] = None
    ) -> Dict:
        """
        Identify if framing members follow standard spacing.

        Args:
            detections: List of detection dictionaries
            category_name: Focus on specific category (e.g., "stud", "joist")

        Returns:
            Dict: Standard spacing analysis
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        # Get spacing data
        spacing_data = self.calculate_spacings(detections)

        if spacing_data["status"] != "success":
            return spacing_data

        spacings = spacing_data["spacings"]

        # Filter by category if specified
        if category_name is not None:
            if category_name not in spacings:
                return {
                    "status": "error",
                    "message": f"Category '{category_name}' not found or insufficient members"
                }

            categories_to_check = {category_name: spacings[category_name]}
        else:
            categories_to_check = spacings

        # Check each category for standard spacing
        standard_results = {}

        for cat, data in categories_to_check.items():
            if "is_standard" in data and data["is_standard"]:
                standard = data["closest_standard"]
                standard_results[cat] = {
                    "is_standard": True,
                    "standard_spacing": standard,
                    "actual_mean": data["mean_spacing"],
                    "confidence": data["confidence"],
                    "unit": data["unit"]
                }
            else:
                standard_results[cat] = {
                    "is_standard": False,
                    "closest_standard": data.get("closest_standard"),
                    "actual_mean": data.get("mean_spacing"),
                    "confidence": data.get("confidence", 0),
                    "unit": data.get("unit")
                }

        return {
            "status": "success",
            "standard_spacing_analysis": standard_results
        }


# models/measurements/dimension_estimator.py
"""
Module for estimating dimensions of framing members.
"""



# models/measurements/path_calculator.py
"""
Module for calculating wiring paths and distances.
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger
from utils.exceptions import MeasurementError

logger = get_logger("path_calculator")

class PathCalculator:
    """
    Class for calculating wiring paths and distances.
    """

    def __init__(self, reference_scale: ReferenceScale):
        """
        Initialize the path calculator.

        Args:
            reference_scale: Reference scale for conversions
        """
        self.reference_scale = reference_scale

    def calculate_path(
        self,
        path_points: List[Tuple[int, int]],
        image: Optional[np.ndarray] = None,
        drill_points: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Calculate the total distance of a wiring path.

        Args:
            path_points: List of (x, y) points defining the path
            image: Original image (optional)
            drill_points: List of points where drilling is required

        Returns:
            Dict: Path analysis results
        """
        if not self.reference_scale.is_calibrated():
            raise MeasurementError("Reference scale is not calibrated")

        if len(path_points) < 2:
            return {
                "status": "error",
                "message": "At least two points needed for a path",
            }

        # Calculate path segments and distances
        segments = []
        total_pixel_distance = 0
        total_real_distance = 0

        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i+1]

            # Calculate segment distance in pixels
            pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Convert to real distance
            real_distance = self.reference_scale.pixels_to_real_distance(pixel_distance)

            # Update totals
            total_pixel_distance += pixel_distance
            total_real_distance += real_distance

            # Store segment information
            segments.append({
                "start": (x1, y1),
                "end": (x2, y2),
                "pixel_distance": pixel_distance,
                "real_distance": real_distance,
                "unit": self.reference_scale.get_unit()
            })

        # Process drill points
        processed_drill_points = []

        if drill_points:
            for point in drill_points:
                processed_drill_points.append({
                    "position": point,
                    "requires_drilling": True
                })

        # Round values for cleaner output
        total_real_distance_rounded = round(total_real_distance, 2)

        # Convert to feet if in inches and distance is large
        display_distance = total_real_distance_rounded
        display_unit = self.reference_scale.get_unit()

        if display_unit == "inches" and total_real_distance_rounded > 24:
            display_distance = total_real_distance_rounded / 12
            display_unit = "feet"

        # Return results
        return {
            "status": "success",
            "path_segments": segments,
            "total_distance": total_real_distance_rounded,
            "display_distance": round(display_distance, 2),
            "display_unit": display_unit,
            "unit": self.reference_scale.get_unit(),
            "drill_points": processed_drill_points,
            "drill_count": len(processed_drill_points)
        }

    def identify_drill_points(
        self,
        path_points: List[Tuple[int, int]],
        detections: List[Dict],
        image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Identify points where the path intersects with framing members.

        Args:
            path_points: List of (x, y) points defining the path
            detections: List of detection dictionaries
            image: Original image (optional)

        Returns:
            List[Dict]: List of drill points
        """
        if len(path_points) < 2:
            return []

        drill_points = []

        # Process each path segment
        for i in range(len(path_points) - 1):
            start_x, start_y = path_points[i]
            end_x, end_y = path_points[i+1]

            # Check intersection with each framing member
            for det in detections:
                # Skip non-framing categories
                category = det['category_name']
                if category not in ['stud', 'joist', 'rafter', 'beam', 'plate', 'header']:
                    continue

                # Get bounding box
                bbox = det['bbox']
                x, y, w, h = bbox

                # Check if segment intersects the bounding box
                if self._segment_intersects_box(start_x, start_y, end_x, end_y, x, y, w, h):
                    # Calculate intersection point
                    intersection = self._get_segment_box_intersection(
                        start_x, start_y, end_x, end_y, x, y, w, h
                    )

                    if intersection:
                        # Determine drill difficulty based on member type and size
                        difficulty = self._calculate_drill_difficulty(det)

                        drill_points.append({
                            "position": intersection,
                            "requires_drilling": True,
                            "category": category,
                            "difficulty": difficulty
                        })

        return drill_points

    def _segment_intersects_box(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        box_x: float,
        box_y: float,
        box_width: float,
        box_height: float
    ) -> bool:
        """
        Check if a line segment intersects with a bounding box.

        Args:
            start_x, start_y: Start point of segment
            end_x, end_y: End point of segment
            box_x, box_y, box_width, box_height: Bounding box

        Returns:
            bool: True if the segment intersects the box
        """
        # Define box corners
        left = box_x
        right = box_x + box_width
        top = box_y
        bottom = box_y + box_height

        # Check if either endpoint is inside the box
        if (left <= start_x <= right and top <= start_y <= bottom) or \
           (left <= end_x <= right and top <= end_y <= bottom):
            return True

        # Check if line segment intersects any of the box edges
        edges = [
            (left, top, right, top),      # Top edge
            (right, top, right, bottom),    # Right edge
            (left, bottom, right, bottom),   # Bottom edge
            (left, top, left, bottom)     # Left edge
        ]

        for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
            if self._line_segments_intersect(
                start_x, start_y, end_x, end_y,
                edge_x1, edge_y1, edge_x2, edge_y2
            ):
                return True

        return False

    def _line_segments_intersect(
        self,
        a_x1: float, a_y1: float, a_x2: float, a_y2: float,
        b_x1: float, b_y1: float, b_x2: float, b_y2: float
    ) -> bool:
        """
        Check if two line segments intersect.

        Args:
            a_x1, a_y1, a_x2, a_y2: First line segment
            b_x1, b_y1, b_x2, b_y2: Second line segment

        Returns:
            bool: True if the segments intersect
        """
        # Calculate the direction vectors
        r = (a_x2 - a_x1, a_y2 - a_y1)
        s = (b_x2 - b_x1, b_y2 - b_y1)

        # Calculate the cross product (r × s)
        rxs = r[0] * s[1] - r[1] * s[0]

        # If r × s = 0, the lines are collinear or parallel
        if abs(rxs) < 1e-8:
            return False

        # Calculate t and u parameters
        q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

        # Check if intersection point is within both segments
        return 0 <= t <= 1 and 0 <= u <= 1

    def _get_segment_box_intersection(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        box_x: float,
        box_y: float,
        box_width: float,
        box_height: float
    ) -> Optional[Tuple[float, float]]:
        """
        Get the intersection point of a line segment with a box.

        Args:
            start_x, start_y: Start point of segment
            end_x, end_y: End point of segment
            box_x, box_y, box_width, box_height: Bounding box

        Returns:
            Optional[Tuple[float, float]]: Intersection point or None
        """
        # Define box corners
        left = box_x
        right = box_x + box_width
        top = box_y
        bottom = box_y + box_height

        # If start point is inside the box, use it
        if left <= start_x <= right and top <= start_y <= bottom:
            return (start_x, start_y)

        # If end point is inside the box, use it
        if left <= end_x <= right and top <= end_y <= bottom:
            return (end_x, end_y)

        # Check intersections with box edges
        edges = [
            (left, top, right, top),      # Top edge
            (right, top, right, bottom),    # Right edge
            (left, bottom, right, bottom),   # Bottom edge
            (left, top, left, bottom)     # Left edge
        ]

        for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
            if self._line_segments_intersect(
                start_x, start_y, end_x, end_y,
                edge_x1, edge_y1, edge_x2, edge_y2
            ):
                # Calculate intersection point
                intersection = self._calculate_intersection_point(
                    start_x, start_y, end_x, end_y,
                    edge_x1, edge_y1, edge_x2, edge_y2
                )

                if intersection:
                    return intersection

        return None

    def _calculate_intersection_point(
        self,
        a_x1: float, a_y1: float, a_x2: float, a_y2: float,
        b_x1: float, b_y1: float, b_x2: float, b_y2: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate the intersection point of two line segments.

        Args:
            a_x1, a_y1, a_x2, a_y2: First line segment
            b_x1, b_y1, b_x2, b_y2: Second line segment

        Returns:
            Optional[Tuple[float, float]]: Intersection point or None
        """
        # Calculate the direction vectors
        r = (a_x2 - a_x1, a_y2 - a_y1)
        s = (b_x2 - b_x1, b_y2 - b_y1)

        # Calculate the cross product (r × s)
        rxs = r[0] * s[1] - r[1] * s[0]

        # If r × s = 0, the lines are collinear or parallel
        if abs(rxs) < 1e-8:
            return None

        # Calculate t parameter
        q_minus_p = (b_x1 - a_x1, b_y1 - a_y1)
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / rxs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / rxs

        # Check if intersection point is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix = a_x1 + t * r[0]
            iy = a_y1 + t * r[1]
            return (ix, iy)

        return None

    def _calculate_drill_difficulty(self, detection: Dict) -> str:
        """
        Estimate the difficulty of drilling through a detected member.

        Args:
            detection: Detection dictionary

        Returns:
            str: Difficulty level ("easy", "moderate", "difficult")
        """
        category = detection['category_name']

        # Estimate based on category
        if category in ['stud', 'plate']:
            base_difficulty = "easy"
        elif category in ['joist', 'rafter']:
            base_difficulty = "moderate"
        elif category in ['beam', 'header']:
            base_difficulty = "difficult"
        else:
            base_difficulty = "moderate"

        # Adjust based on member size if available
        if 'dimensions' in detection:
            thickness = detection['dimensions'].get('thickness', 0)

            # Convert to inches for standard comparison
            if self.reference_scale.get_unit() != "inches":
                thickness_inches = self.reference_scale.convert_units(
                    thickness, self.reference_scale.get_unit(), "inches"
                )
            else:
                thickness_inches = thickness

            # Adjust difficulty based on thickness
            if thickness_inches > 3.0: # Thick member
                if base_difficulty == "easy":
                    base_difficulty = "moderate"
                elif base_difficulty == "moderate":
                    base_difficulty = "difficult"
            elif thickness_inches < 1.0: # Thin member
                if base_difficulty == "difficult":
                    base_difficulty = "moderate"
                elif base_difficulty == "moderate":
                    base_difficulty = "easy"

        return base_difficulty

    def estimate_drilling_time(
        self,
        drill_points: List[Dict],
        drill_speed: str = "normal"
    ) -> Dict:
        """
        Estimate the time required for drilling through framing members.

        Args:
            drill_points: List of drill points
            drill_speed: Drilling speed ("slow", "normal", "fast")

        Returns:
            Dict: Time estimates
        """
        if not drill_points:
            return {
                "total_time_minutes": 0,
                "drill_points": 0,
                "average_time_per_point": 0
            }

        # Base time per difficulty level (in minutes)
        time_factors = {
            "easy": {"slow": 5, "normal": 3, "fast": 2},
            "moderate": {"slow": 8, "normal": 5, "fast": 3},
            "difficult": {"slow": 12, "normal": 8, "fast": 5}
        }

        total_time = 0

        for point in drill_points:
            difficulty = point.get("difficulty", "moderate")
            time_per_drill = time_factors.get(difficulty, time_factors["moderate"])[drill_speed]
            total_time += time_per_drill

        return {
            "total_time_minutes": total_time,
            "drill_points": len(drill_points),
            "average_time_per_point": total_time / len(drill_points)
        }


# models/measurements/visualization.py
"""
Visualization utilities for measurement estimations.
"""

import numpy as np
import cv2
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from models.measurements.reference_scale import ReferenceScale
from utils.logger import get_logger

logger = get_logger("visualization")

def visualize_measurements(
    image: np.ndarray,
    measurement_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_dimensions: bool = True,
    show_spacings: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize measurement results on an image.

    Args:
        image: Input image
        measurement_result: Measurement analysis results
        output_path: Path to save visualization
        show_dimensions: Whether to show member dimensions
        show_spacings: Whether to show spacing measurements
        line_thickness: Line thickness for drawings
        font_scale: Font scale for text

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # White

    # Get measurements data
    if 'measurements' not in measurement_result:
        logger.warning("No measurements data in result")
        return vis_img

    measurements = measurement_result['measurements']
    unit = measurements.get('unit', 'inches')

    # Visualize dimensions for each member
    if show_dimensions and 'dimensions' in measurements:
        for cat_name, dim_data in measurements['dimensions'].items():
            # Set color based on category
            if cat_name == 'stud':
                color = (0, 255, 0)  # Green
            elif cat_name == 'joist':
                color = (255, 0, 0)  # Blue
            elif cat_name == 'rafter':
                color = (0, 0, 255) # Red
            elif cat_name == 'beam':
                color = (0, 255, 255) # Yellow
            else:
                color = (255, 255, 0) # Cyan

            # Draw dimensions for each member
            for member in dim_data.get('dimensions', []):
                bbox = member.get('bbox')
                if not bbox:
                    continue

                x, y, w, h = bbox

                # Draw bounding box
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), color, line_thickness)

                # Draw dimension labels
                size_name = member.get('standard_size', 'unknown')
                thickness = round(member.get('thickness', 0), 1)
                depth = round(member.get('depth', 0), 1)

                if size_name != 'custom':
                    label = f"{size_name} ({thickness} x {depth} {unit})"
                else:
                    label = f"{thickness} x {depth} {unit}"

                # Draw label background
                text_size = cv2.getTextSize(label, font, font_scale, line_thickness)[0]
                cv2.rectangle(
                    vis_img,
                    (int(x), int(y - text_size[1] - 5)),
                    (int(x + text_size[0]), int(y)),
                    color,
                    -1
                )

                # Draw label
                cv2.putText(
                    vis_img,
                    label,
                    (int(x), int(y - 5)),
                    font,
                    font_scale,
                    text_color,
                    1
                )

    # Visualize spacings between members
    if show_spacings and 'spacings' in measurements:
        for cat_name, spacing_data in measurements['spacings'].items():
            # Set color based on category
            if cat_name == 'stud':
                color = (0, 165, 255) # Orange
            elif cat_name == 'joist':
                color = (255, 0, 255) # Magenta
            elif cat_name == 'rafter':
                color = (128, 0, 128) # Purple
            else:
                color = (128, 128, 0) # Olive

            # Get center points and spacings
            center_points = spacing_data.get('center_points', [])
            spacings = spacing_data.get('spacings', [])
            orientation = spacing_data.get('orientation', 'mixed')

            # Draw lines between centers
            if len(center_points) < 2:
                continue

            for i in range(len(center_points) - 1):
                p1 = (int(center_points[i][0]), int(center_points[i][1]))
                p2 = (int(center_points[i+1][0]), int(center_points[i+1][1]))

                # Draw line between centers
                cv2.line(vis_img, p1, p2, color, line_thickness)

                # Draw spacing measurement
                if i < len(spacings):
                    spacing_value = round(spacings[i], 1)
                    spacing_label = f"{spacing_value} {unit}"

                    # Calculate label position
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2

                    # Adjust position based on orientation
                    if orientation == "vertical":
                        text_pos = (mid_x, mid_y - 10)
                    elif orientation == "horizontal":
                        text_pos = (mid_x + 10, mid_y)
                    else:
                        text_pos = (mid_x, mid_y - 10)

                    # Draw label background
                    text_size = cv2.getTextSize(spacing_label, font, font_scale, 1)[0]
                    cv2.rectangle(
                        vis_img,
                        (int(text_pos[0]), int(text_pos[1] - text_size[1])),
                        (int(text_pos[0] + text_size[0]), int(text_pos[1] + 5)),
                        color,
                        -1
                    )

                    # Draw label
                    cv2.putText(
                        vis_img,
                        spacing_label,
                        text_pos,
                        font,
                        font_scale,
                        text_color,
                        1
                    )

    # Add overall information
    confidence = measurement_result.get('confidence', 0)
    confidence_text = f"Confidence: {confidence:.2f}"

    cv2.putText(
        vis_img,
        confidence_text,
        (10, 30),
        font,
        0.7,
        (0, 255, 255),
        2
    )

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved measurement visualization to {output_path}")

    return vis_img

def visualize_wiring_path(
    image: np.ndarray,
    path_result: Dict,
    output_path: Optional[Union[str, Path]] = None,
    show_distance: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Visualize a wiring path on an image.

    Args:
        image: Input image
        path_result: Path analysis results
        output_path: Path to save visualization
        show_distance: Whether to show distance measurements
        line_thickness: Line thickness for drawings

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    path_color = (0, 255, 255)  # Yellow
    drill_color = (0, 0, 255)   # Red
    text_color = (255, 255, 255)  # White

    # Draw path segments
    segments = path_result.get('path_segments', [])
    for segment in segments:
        start = segment.get('start')
        end = segment.get('end')

        if not start or not end:
            continue

        # Draw line segment
        cv2.line(
            vis_img,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            path_color,
            line_thickness
        )

        # Draw distance if requested
        if show_distance:
            real_distance = segment.get('real_distance', 0)
            unit = segment.get('unit', 'inches')
            distance_label = f"{real_distance:.1f} {unit}"

            # Calculate label position
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2

            # Draw label background
            text_size = cv2.getTextSize(distance_label, font, 0.5, 1)[0]
            cv2.rectangle(
                vis_img,
                (int(mid_x), int(mid_y - text_size[1])),
                (int(mid_x + text_size[0]), int(mid_y + 5)),
                path_color,
                -1
            )

            # Draw label
            cv2.putText(
                vis_img,
                distance_label,
                (int(mid_x), int(mid_y)),
                font,
                0.5,
                text_color,
                1
            )

    # Draw drill points
    drill_points = path_result.get('drill_points', [])
    for point in drill_points:
        position = point.get('position')
        difficulty = point.get('difficulty', 'moderate')

        if not position:
            continue

        # Adjust color based on difficulty
        if difficulty == 'easy':
            point_color = (0, 255, 0) # Green
        elif difficulty == 'moderate':
            point_color = (0, 165, 255) # Orange
        else: # difficult
            point_color = (0, 0, 255)   # Red

        # Draw drill point
        cv2.circle(
            vis_img,
            (int(position[0]), int(position[1])),
            8,
            point_color,
            -1
        )

        # Draw label
        cv2.putText(
            vis_img,
            difficulty,
            (int(position[0]) + 10, int(position[1])),
            font,
            0.5,
            point_color,
            2
        )

    # Add total distance at the top
    total_distance = path_result.get('display_distance', 0)
    display_unit = path_result.get('display_unit', 'inches')
    drill_count = path_result.get('drill_count', 0)

    total_text = f"Total: {total_distance:.1f} {display_unit}"
    drill_text = f"Drill points: {drill_count}"

    cv2.putText(vis_img, total_text, (10, 30), font, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_img, drill_text, (10, 60), font, 0.7, (0, 0, 255), 2)

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved wiring path visualization to {output_path}")

    return vis_img

def visualize_scale_calibration(
    image: np.ndarray,
    calibration_data: Dict,
    output_path: Optional[Union[str, Path]] = None,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Visualize scale calibration on an image.

    Args:
        image: Input image
        calibration_data: Calibration data
        output_path: Path to save visualization
        line_thickness: Line thickness for drawings

    Returns:
        np.ndarray: Visualization image
    """
    # Create a copy of the image
    vis_img = image.copy()

    # Set default font and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    calib_color = (0, 255, 0)  # Green
    text_color = (255, 255, 255)  # White

    method = calibration_data.get('method', 'unknown')

    # Draw calibration visualization based on method
    if method == 'point_distance':
        # Draw reference line between points
        reference_data = calibration_data.get('reference_data', {})
        points = reference_data.get('points', [])

        if len(points) >= 2:
            p1 = (int(points[0][0]), int(points[0][1]))
            p2 = (int(points[1][0]), int(points[1][1]))

            # Draw reference line
            cv2.line(vis_img, p1, p2, calib_color, line_thickness)

            # Draw endpoints
            cv2.circle(vis_img, p1, 5, calib_color, -1)
            cv2.circle(vis_img, p2, 5, calib_color, -1)

            # Draw distance label
            known_distance = reference_data.get('known_distance', 0)
            unit = calibration_data.get('unit', 'inches')
            label = f"{known_distance} {unit}"

            # Calculate label position
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            # Draw label background
            text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
            cv2.rectangle(
                vis_img,
                (mid_x, mid_y - text_size[1] - 5),
                (mid_x + text_size[0], mid_y),
                calib_color,
                -1
            )

            # Draw label
            cv2.putText(
                vis_img,
                label,
                (mid_x, mid_y - 5),
                font,
                0.6,
                text_color,
                1
            )

    elif method == 'object_dimensions':
        # Draw reference object box
        reference_data = calibration_data.get('reference_data', {})
        object_bbox = reference_data.get('object_bbox', [])

        if len(object_bbox) == 4:
            x, y, w, h = object_bbox

            # Draw bounding box
            cv2.rectangle(
                vis_img,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                calib_color,
                line_thickness
            )

            # Draw dimension labels
            dimensions = reference_data.get('object_dimensions', [0, 0])
            unit = calibration_data.get('unit', 'inches')

            width_label = f"Width: {dimensions[0]} {unit}"
            height_label = f"Height: {dimensions[1]} {unit}"

            # Draw width label
            cv2.putText(
                vis_img,
                width_label,
                (int(x), int(y - 10)),
                font,
                0.6,
                calib_color,
                2
            )

            # Draw height label
            cv2.putText(
                vis_img,
                height_label,
                (int(x + w + 5), int(y + h // 2)),
                font,
                0.6,
                calib_color,
                2
            )

    # Add calibration information
    pixels_per_unit = calibration_data.get('pixels_per_unit', 0)
    unit = calibration_data.get('unit', 'inches')
    confidence = calibration_data.get('confidence', 0)

    info_text1 = f"Scale: {pixels_per_unit:.2f} pixels per {unit}"
    info_text2 = f"Confidence: {confidence:.2f}"

    cv2.putText(vis_img, info_text1, (10, 30), font, 0.7, calib_color, 2)
    cv2.putText(vis_img, info_text2, (10, 60), font, 0.7, calib_color, 2)

    # Save visualization if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"Saved calibration visualization to {output_path}")

    return vis_img








#Now I'll implement the end-to-end integration code to connect all the components of the electrician time estimation application together. This will involve creating code that:

#1. Connects the frontend UI to the backend API
#2. Ensures proper data flow between detection, estimation, and display components
#3. Implements error handling and recovery
#4. Adds progress indicators for long-running processes
#5. Creates a configuration system for application settings

#Let me start by implementing the backend API routes that will serve the frontend:


# api/routes/estimation.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import tempfile
import json
import os
import shutil
from typing import Dict, List, Optional, Any

from models.detection import FramingDetector, detect_framing
from models.measurements import MeasurementEstimator
from utils.logger import get_logger
from utils.exceptions import EstimatorBaseException
from config.settings import (
    MODEL_WEIGHTS_DIR, IMAGES_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    COMPLEXITY_THRESHOLDS, BASE_TIME_MINUTES
)

logger = get_logger("api_estimation")
router = APIRouter()

# Task storage for background tasks
TASKS = {}

# Initialize models
detector = None
estimator = None

def load_models():
    """
    Lazy-load models when needed.
    """
    global detector, estimator
    
    if detector is None:
        # Find best available model weights
        model_paths = list(MODEL_WEIGHTS_DIR.glob("framing_detector_*.pt"))
        if model_paths:
            model_path = sorted(model_paths)[-1]  # Use most recent model
            logger.info(f"Loading detection model from {model_path}")
            detector = FramingDetector.from_checkpoint(model_path)
        else:
            logger.warning("No detection model weights found, using pretrained model")
            detector = FramingDetector(pretrained=True)
    
    if estimator is None:
        estimator = MeasurementEstimator()
    
    return detector, estimator

def process_image_task(task_id: str, image_path: Path):
    """
    Background task to process an image.
    """
    try:
        # Update task status
        TASKS[task_id]["status"] = "processing"
        
        # Load the models
        detector, estimator = load_models()
        
        # Run detection
        detection_result = detect_framing(detector, str(image_path))
        TASKS[task_id]["progress"] = 40
        
        # If we have a stored calibration, use it
        calibration_data = TASKS[task_id].get("calibration_data")
        if calibration_data:
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        else:
            # Try to auto-calibrate using a known framing member
            if detection_result["detections"]:
                # Find a likely stud or joist for calibration
                for det in detection_result["detections"]:
                    if det["category_name"] in ["stud", "joist"] and det["confidence"] > 0.8:
                        # Use standard framing dimensions for calibration
                        # 2x4 stud is typically 1.5 x 3.5 inches
                        try:
                            estimator.calibrate_from_known_object(
                                detection_result["image"],
                                det["bbox"],
                                (1.5, 3.5)  # 2x4 nominal dimensions
                            )
                            TASKS[task_id]["calibration_data"] = estimator.reference_scale.get_calibration_data()
                            break
                        except Exception as e:
                            logger.warning(f"Auto-calibration failed: {str(e)}")
        
        TASKS[task_id]["progress"] = 60
        
        # Run measurements
        measurement_result = estimator.analyze_framing_measurements(
            detection_result, calibration_check=False
        )
        TASKS[task_id]["progress"] = 80
        
        # Estimate time required based on complexity
        time_estimate = estimate_time(detection_result, measurement_result)
        
        # Save results
        results = {
            "detections": detection_result["detections"],
            "measurements": measurement_result["measurements"],
            "time_estimate": time_estimate
        }
        
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["results"] = results
        
        # Clean up - keep for 1 hour in a real app
        # del TASKS[task_id]
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)

def estimate_time(detection_result: Dict, measurement_result: Dict) -> Dict:
    """
    Estimate time required for electrical work based on detection and measurements.
    """
    # Count different framing members
    member_counts = {}
    for det in detection_result["detections"]:
        cat = det["category_name"]
        member_counts[cat] = member_counts.get(cat, 0) + 1
    
    # Calculate complexity factors
    stud_count = member_counts.get("stud", 0)
    joist_count = member_counts.get("joist", 0)
    obstacle_count = member_counts.get("obstacle", 0) + member_counts.get("plumbing", 0)
    electrical_box_count = member_counts.get("electrical_box", 0)
    
    # Determine spacing complexity (irregular spacing is more complex)
    spacing_complexity = 0
    spacings = measurement_result.get("measurements", {}).get("spacing", {}).get("spacings", {})
    
    for cat, spacing_data in spacings.items():
        if not spacing_data.get("is_standard", True):
            spacing_complexity += 0.3
    
    # Calculate overall complexity score
    complexity_score = (
        stud_count * 0.05 +
        joist_count * 0.05 +
        obstacle_count * 0.2 +
        electrical_box_count * 0.1 +
        spacing_complexity
    )
    
    # Map to complexity level
    if complexity_score <= COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"
    elif complexity_score <= COMPLEXITY_THRESHOLDS["moderate"]:
        complexity = "moderate"
    else:
        complexity = "complex"
    
    # Calculate time estimate
    base_time = BASE_TIME_MINUTES[complexity]
    total_time = base_time + (stud_count * 0.5) + (obstacle_count * 1.5) + (electrical_box_count * 2)
    
    return {
        "complexity": complexity,
        "complexity_score": complexity_score,
        "estimated_minutes": round(total_time),
        "factors": {
            "stud_count": stud_count,
            "joist_count": joist_count,
            "obstacle_count": obstacle_count,
            "electrical_box_count": electrical_box_count,
            "spacing_complexity": spacing_complexity
        }
    }

@router.post("/estimate")
async def estimate_from_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    calibration: Optional[str] = Form(None)
):
    """
    Submit an image for time estimation (processes in background).
    Returns a task ID for checking status.
    """
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Create temp directory for this task
    task_dir = IMAGES_DIR / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    image_path = task_dir / f"input{Path(file.filename).suffix}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse calibration data if provided
    calibration_data = None
    if calibration:
        try:
            calibration_data = json.loads(calibration)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid calibration data format")
    
    # Create task entry
    TASKS[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "image_path": str(image_path),
        "calibration_data": calibration_data
    }
    
    # Start background processing
    background_tasks.add_task(process_image_task, task_id, image_path)
    
    return {"task_id": task_id, "status": "pending"}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a submitted task.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id].copy()
    
    # Don't return the image path in the response
    if "image_path" in task:
        del task["image_path"]
    
    return task

@router.post("/calibrate")
async def calibrate_from_image(
    file: UploadFile = File(...),
    reference_type: str = Form(...),
    reference_data: str = Form(...)
):
    """
    Calibrate the measurement system using a reference object or points.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Load the image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Initialize the estimator
        _, estimator = load_models()
        
        # Parse reference data
        ref_data = json.loads(reference_data)
        
        # Perform calibration based on reference type
        if reference_type == "points":
            if "points" not in ref_data or "distance" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing points or distance")
            
            points = ref_data["points"]
            distance = float(ref_data["distance"])
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_reference(
                image, points, distance, units
            )
        
        elif reference_type == "object":
            if "bbox" not in ref_data or "dimensions" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing bbox or dimensions")
            
            bbox = ref_data["bbox"]
            dimensions = ref_data["dimensions"]
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_known_object(
                image, bbox, dimensions, units
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid reference type")
        
        # Clean up
        os.unlink(temp_path)
        
        return calibration
        
    except EstimatorBaseException as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        logger.error(f"Calibration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during calibration")

@router.post("/path")
async def estimate_wiring_path(
    file: UploadFile = File(...),
    path_points: str = Form(...),
    calibration: Optional[str] = Form(None)
):
    """
    Estimate wiring path distances and drilling points.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Load the image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Load models
        detector, estimator = load_models()
        
        # Parse path points
        points = json.loads(path_points)
        
        # Parse calibration data if provided
        if calibration:
            calibration_data = json.loads(calibration)
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        
        # Run detection
        detection_result = detect_framing(detector, image)
        
        # Estimate wiring path
        path_result = estimator.estimate_wiring_path(
            detection_result=detection_result,
            path_points=points
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return path_result
        
    except EstimatorBaseException as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        logger.error(f"Path estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during path estimation")


#Now, let's implement the frontend part with Electron. I'll start with the main Electron app file:

javascript
// frontend/electron/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const url = require('url');

// Keep a global reference of the window object
let mainWindow;

// Configuration
const config = {
  apiBaseUrl: 'http://localhost:8000/api',
  settings: {
    autoCalibrate: true,
    confidenceThreshold: 0.25,
    showMeasurements: true
  }
};

// Load config from file if exists
try {
  const userDataPath = app.getPath('userData');
  const configPath = path.join(userDataPath, 'config.json');
  if (fs.existsSync(configPath)) {
    const savedConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    config.settings = {...config.settings, ...savedConfig.settings};
    if (savedConfig.apiBaseUrl) {
      config.apiBaseUrl = savedConfig.apiBaseUrl;
    }
  }
} catch (error) {
  console.error('Error loading config:', error);
}

// Save config to file
function saveConfig() {
  try {
    const userDataPath = app.getPath('userData');
    const configPath = path.join(userDataPath, 'config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  } catch (error) {
    console.error('Error saving config:', error);
  }
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true, // Be careful with this in production!
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Load the index.html of the app
  mainWindow.loadFile(path.join(__dirname, 'src/index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  // Emitted when the window is closed
  mainWindow.on('closed', function() {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed
app.on('window-all-closed', function() {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function() {
  if (mainWindow === null) createWindow();
});

// IPC handlers
ipcMain.handle('select-image', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }]
  });
  
  if (canceled) return null;
  return filePaths[0];
});

ipcMain.handle('get-config', () => {
  return config;
});

ipcMain.handle('save-config', (event, newSettings) => {
  config.settings = {...config.settings, ...newSettings};
  config.apiBaseUrl = newSettings.apiBaseUrl || config.apiBaseUrl;
  saveConfig();
  return config;
});

ipcMain.handle('submit-image', async (event, imagePath, calibrationData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    
    if (calibrationData) {
      formData.append('calibration', JSON.stringify(calibrationData));
    }
    
    const response = await axios.post(
      `${config.apiBaseUrl}/estimate`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000 // 30 seconds timeout
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error submitting image:', error);
    throw new Error(error.response?.data?.detail || 'Failed to submit image');
  }
});

ipcMain.handle('check-task', async (event, taskId) => {
  try {
    const response = await axios.get(
      `${config.apiBaseUrl}/tasks/${taskId}`,
      { timeout: 10000 }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error checking task:', error);
    throw new Error(error.response?.data?.detail || 'Failed to check task status');
  }
});

ipcMain.handle('calibrate-image', async (event, imagePath, referenceType, referenceData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    formData.append('reference_type', referenceType);
    formData.append('reference_data', JSON.stringify(referenceData));
    
    const response = await axios.post(
      `${config.apiBaseUrl}/calibrate`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error calibrating image:', error);
    throw new Error(error.response?.data?.detail || 'Failed to calibrate image');
  }
});

ipcMain.handle('estimate-path', async (event, imagePath, pathPoints, calibrationData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    formData.append('path_points', JSON.stringify(pathPoints));
    
    if (calibrationData) {
      formData.append('calibration', JSON.stringify(calibrationData));
    }
    
    const response = await axios.post(
      `${config.apiBaseUrl}/path`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error estimating path:', error);
    throw new Error(error.response?.data?.detail || 'Failed to estimate wiring path');
  }
});


#Now, let's define a preload script that will expose the IPC functions to our renderer process:

javascript
// frontend/electron/preload.js
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    selectImage: () => ipcRenderer.invoke('select-image'),
    getConfig: () => ipcRenderer.invoke('get-config'),
    saveConfig: (settings) => ipcRenderer.invoke('save-config', settings),
    submitImage: (imagePath, calibrationData) => ipcRenderer.invoke('submit-image', imagePath, calibrationData),
    checkTask: (taskId) => ipcRenderer.invoke('check-task', taskId),
    calibrateImage: (imagePath, referenceType, referenceData) => 
      ipcRenderer.invoke('calibrate-image', imagePath, referenceType, referenceData),
    estimatePath: (imagePath, pathPoints, calibrationData) => 
      ipcRenderer.invoke('estimate-path', imagePath, pathPoints, calibrationData)
  }
);


#Next, let's create the main HTML file for the frontend:

html
<!-- frontend/electron/src/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electrician Time Estimator</title>
    <link rel="stylesheet" href="styles/main.css">
</head>
<body>
    <header>
        <h1>Electrician Time Estimator</h1>
        <div class="toolbar">
            <button id="selectImageBtn">Select Image</button>
            <button id="calibrateBtn">Calibrate</button>
            <button id="drawPathBtn">Draw Wiring Path</button>
            <button id="settingsBtn">Settings</button>
        </div>
    </header>

    <main>
        <div id="imageContainer">
            <div id="placeholder">
                <p>Select an image to analyze</p>
            </div>
            <div id="canvasContainer" style="display: none;">
                <canvas id="mainCanvas"></canvas>
            </div>
            <div id="progress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <p class="progress-text">Processing...</p>
            </div>
        </div>

        <div id="resultsPanel">
            <div class="panel-header">
                <h2>Results</h2>
                <button id="exportBtn" disabled>Export Report</button>
            </div>
            <div id="resultsContent">
                <p class="placeholder">No results yet. Process an image to see time estimates.</p>
            </div>
        </div>
    </main>

    <div id="calibrateModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Calibrate Measurements</h2>
            <p>Select a calibration method:</p>
            
            <div class="calibration-options">
                <button class="calibrate-option" data-method="known-object">Use Known Object</button>
                <button class="calibrate-option" data-method="reference-points">Use Reference Points</button>
            </div>
            
            <div id="calibrateKnownObject" class="calibration-method">
                <p>Select a framing member by drawing a box around it:</p>
                <button id="drawBoxBtn">Draw Box</button>
                <div>
                    <label>Object Type:</label>
                    <select id="objectType">
                        <option value="2x4">2x4 Stud (1.5" x 3.5")</option>
                        <option value="2x6">2x6 (1.5" x 5.5")</option>
                        <option value="2x8">2x8 (1.5" x 7.25")</option>
                        <option value="4x4">4x4 Post (3.5" x 3.5")</option>
                        <option value="custom">Custom Dimensions</option>
                    </select>
                </div>
                <div id="customDimensions" style="display: none;">
                    <label>Width (inches): <input id="customWidth" type="number" step="0.25" min="0.5" value="1.5"></label>
                    <label>Height (inches): <input id="customHeight" type="number" step="0.25" min="0.5" value="3.5"></label>
                </div>
                <button id="submitKnownObjectBtn" disabled>Submit Calibration</button>
            </div>
            
            <div id="calibrateReferencePoints" class="calibration-method">
                <p>Click to place two points at a known distance:</p>
                <div>
                    <label>Distance between points:</label>
                    <input id="referenceDistance" type="number" step="0.25" min="0.5" value="16">
                    <select id="distanceUnit">
                        <option value="inches">inches</option>
                        <option value="feet">feet</option>
                        <option value="cm">centimeters</option>
                        <option value="m">meters</option>
                    </select>
                </div>
                <button id="submitPointsBtn" disabled>Submit Calibration</button>
            </div>
        </div>
    </div>

    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Settings</h2>
            
            <div class="settings-form">
                <div class="form-group">
                    <label for="apiUrl">API URL:</label>
                    <input type="text" id="apiUrl" placeholder="http://localhost:8000/api">
                </div>
                
                <div class="form-group">
                    <label for="confidenceThreshold">Detection Confidence Threshold:</label>
                    <input type="range" id="confidenceThreshold" min="0.05" max="0.95" step="0.05" value="0.25">
                    <span id="confidenceValue">0.25</span>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="autoCalibrate" checked>
                    <label for="autoCalibrate">Auto-calibrate using detected framing members</label>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="showMeasurements" checked>
                    <label for="showMeasurements">Show measurements on detection</label>
                </div>
                
                <button id="saveSettingsBtn">Save Settings</button>
            </div>
        </div>
    </div>

    <div id="pathModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Draw Wiring Path</h2>
            <p>Click on the image to place path points:</p>
            
            <div class="path-controls">
                <button id="clearPathBtn">Clear Path</button>
                <button id="undoPointBtn">Undo Last Point</button>
                <button id="calculatePathBtn" disabled>Calculate Path</button>
            </div>
            
            <div id="pathResults" style="display: none;">
                <h3>Wiring Path Results</h3>
                <div id="pathResultsContent"></div>
            </div>
        </div>
    </div>

    <script src="scripts/main.js"></script>
</body>
</html>


#Now, let's add some CSS styles:

css
/* frontend/electron/src/styles/main.css */
:root {
    --primary-color: #3498db;
    --primary-dark: #2980b9;
    --secondary-color: #2ecc71;
    --accent-color: #e74c3c;
    --text-color: #333;
    --light-gray: #f4f4f4;
    --medium-gray: #ddd;
    --dark-gray: #888;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: var(--text-color);
    line-height: 1.6;
    background-color: #f9f9f9;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

header {
    background-color: var(--primary-color);
    color: white;
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    font-size: 1.5rem;
}

main {
    display: flex;
    flex: 1;
    padding: 1rem;
    gap: 1rem;
    overflow: hidden;
}

button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

button:hover {
    background-color: var(--primary-dark);
}

button:disabled {
    background-color: var(--dark-gray);
    cursor: not-allowed;
}

.toolbar {
    display: flex;
    gap: 0.5rem;
}

#imageContainer {
    flex: 1.5;
    background-color: var(--light-gray);
    border-radius: 4px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
}

#placeholder {
    text-align: center;
    color: var(--dark-gray);
}

#canvasContainer {
    width: 100%;
    height: 100%;
    position: relative;
}

canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

#resultsPanel {
    flex: 1;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.panel-header {
    padding: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--light-gray);
    border-bottom: 1px solid var(--medium-gray);
}

.panel-header h2 {
    font-size: 1.2rem;
    font-weight: 500;
}

#resultsContent {
    padding: 1rem;
    overflow-y: auto;
    flex: 1;
}

.placeholder {
    color: var(--dark-gray);
    text-align: center;
    padding: 2rem 1rem;
}

/* Progress bar styles */
#progress {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-top: 1px solid var(--medium-gray);
}

.progress-bar {
    height: 10px;
    background-color: var(--light-gray);
    border-radius: 5px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--secondary-color);
    width: 0%;
    transition: width 0.3s ease-in-out;
}

.progress-text {
    text-align: center;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    z-index: 100;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background-color: white;
    margin: 10% auto;
    padding: 1.5rem;
    border-radius: 6px;
    width: 60%;
    max-width: 600px;
    position: relative;
}

.close-btn {
    position: absolute;
    top: 0.75rem;
    right: 1rem;
    font-size: 1.5rem;
    cursor: pointer;
}

.calibration-options {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.calibration-method {
    display: none;
    margin-top: 1rem;
    padding: 1rem;
    background-color: var(--light-gray);
    border-radius: 4px;
}

/* Form controls */
.form-group {
    margin-bottom: 1rem;
}

label {
    display: block;
    margin-bottom: 0.3rem;
}

input[type="text"],
input[type="number"],
select {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--medium-gray);
    border-radius: 4px;
}

input[type="range"] {
    width: 90%;
    vertical-align: middle;
}

.checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.checkbox label {
    margin-bottom: 0;
}

.checkbox input {
    width: auto;
}

/* Path drawing */
.path-controls {
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0;
}

/* Results styling */
.result-section {
    margin-bottom: 1.5rem;
}

.result-section h3 {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--light-gray);
}

.result-item {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
}

.detection-item {
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    background-color: var(--light-gray);
    border-radius: 4px;
}

.time-estimate {
    font-size: 1.5rem;
    text-align: center;
    padding: 1rem;
    margin: 1rem 0;
    background-color: var(--light-gray);
    border-radius: 4px;
    font-weight: bold;
}

.complexity-simple {
    color: var(--secondary-color);
}

.complexity-moderate {
    color: orange;
}

.complexity-complex {
    color: var(--accent-color);
}


#Finally, let's implement the main JavaScript file that will handle the UI and interactions:

javascript
// frontend/electron/src/scripts/main.js
// Global state
let state = {
  currentImage: null,
  imageElement: null,
  calibrationData: null,
  taskId: null,
  detectionResults: null,
  pathPoints: [],
  drawingBox: false,
  boxStart: null,
  referencePoints: [],
  isDrawingPath: false,
  settings: {
    apiBaseUrl: 'http://localhost:8000/api',
    confidenceThreshold: 0.25,
    autoCalibrate: true,
    showMeasurements: true
  }
};

// Canvas for image display and interaction
let canvas = document.getElementById('mainCanvas');
let ctx = canvas.getContext('2d');

// UI Elements
const selectImageBtn = document.getElementById('selectImageBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const drawPathBtn = document.getElementById('drawPathBtn');
const settingsBtn = document.getElementById('settingsBtn');
const exportBtn = document.getElementById('exportBtn');
const placeholder = document.getElementById('placeholder');
const canvasContainer = document.getElementById('canvasContainer');
const progressContainer = document.getElementById('progress');
const progressFill = document.querySelector('.progress-fill');
const progressText = document.querySelector('.progress-text');
const resultsContent = document.getElementById('resultsContent');

// Modal elements
const calibrateModal = document.getElementById('calibrateModal');
const settingsModal = document.getElementById('settingsModal');
const pathModal = document.getElementById('pathModal');
const closeButtons = document.querySelectorAll('.close-btn');

// Initialize the app
async function init() {
  // Load settings
  try {
    const config = await window.api.getConfig();
    state.settings = {...state.settings, ...config.settings};
    if (config.apiBaseUrl) {
      state.settings.apiBaseUrl = config.apiBaseUrl;
    }
    
    // Update UI with settings
    document.getElementById('apiUrl').value = state.settings.apiBaseUrl;
    document.getElementById('confidenceThreshold').value = state.settings.confidenceThreshold;
    document.getElementById('confidenceValue').textContent = state.settings.confidenceThreshold;
    document.getElementById('autoCalibrate').checked = state.settings.autoCalibrate;
    document.getElementById('showMeasurements').checked = state.settings.showMeasurements;
  } catch (error) {
    console.error('Error loading settings:', error);
  }
  
  // Set up event listeners
  setupEventListeners();
}

function setupEventListeners() {
  // Main buttons
  selectImageBtn.addEventListener('click', handleSelectImage);
  calibrateBtn.addEventListener('click', () => showModal(calibrateModal));
  drawPathBtn.addEventListener('click', startPathDrawing);
  settingsBtn.addEventListener('click', () => showModal(settingsModal));
  exportBtn.addEventListener('click', exportReport);
  
  // Close modal buttons
  closeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const modal = btn.closest('.modal');
      hideModal(modal);
    });
  });
  
  // Calibration modal
  document.querySelectorAll('.calibrate-option').forEach(btn => {
    btn.addEventListener('click', selectCalibrationMethod);
  });
  
  document.getElementById('objectType').addEventListener('change', handleObjectTypeChange);
  document.getElementById('drawBoxBtn').addEventListener('click', startBoxDrawing);
  document.getElementById('submitKnownObjectBtn').addEventListener('click', submitKnownObjectCalibration);
  document.getElementById('submitPointsBtn').addEventListener('click', submitPointsCalibration);
  
  // Settings modal
  document.getElementById('confidenceThreshold').addEventListener('input', updateConfidenceLabel);
  document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
  
  // Path modal
  document.getElementById('clearPathBtn').addEventListener('click', clearPath);
  document.getElementById('undoPointBtn').addEventListener('click', undoLastPathPoint);
  document.getElementById('calculatePathBtn').addEventListener('click', calculatePath);
  
  // Canvas interactions
  canvas.addEventListener('mousedown', handleCanvasMouseDown);
  canvas.addEventListener('mousemove', handleCanvasMouseMove);
  canvas.addEventListener('mouseup', handleCanvasMouseUp);
}

// Image selection and processing
async function handleSelectImage() {
  try {
    const imagePath = await window.api.selectImage();
    if (!imagePath) return;
    
    // Reset state
    state.currentImage = imagePath;
    state.detectionResults = null;
    state.taskId = null;
    exportBtn.disabled = true;
    
    // Show the image in canvas
    loadImageToCanvas(imagePath);
    
    // Hide placeholder, show canvas
    placeholder.style.display = 'none';
    canvasContainer.style.display = 'block';
    
    // Process the image
    await processImage();
  } catch (error) {
    showError('Failed to select image: ' + error.message);
  }
}

function loadImageToCanvas(imagePath) {
  // Create an image element
  const img = new Image();
  img.onload = function() {
    // Set canvas dimensions to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw image to canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    // Store image element
    state.imageElement = img;
  };
  
  // Set image source to file path (using URL scheme for Electron)
  img.src = 'file://' + imagePath;
}

async function processImage() {
  try {
    // Show progress bar
    showProgress(10, 'Submitting image...');
    
    // Submit the image for processing
    const taskResponse = await window.api.submitImage(
      state.currentImage, 
      state.calibrationData
    );
    
    state.taskId = taskResponse.task_id;
    
    // Poll for task completion
    await pollTaskStatus();
  } catch (error) {
    hideProgress();
    showError('Failed to process image: ' + error.message);
  }
}

async function pollTaskStatus() {
  if (!state.taskId) return;
  
  try {
    let completed = false;
    while (!completed) {
      const taskStatus = await window.api.checkTask(state.taskId);
      
      // Update progress
      showProgress(taskStatus.progress || 10, `${taskStatus.status}...`);
      
      if (taskStatus.status === 'completed') {
        // Process is complete
        state.detectionResults = taskStatus.results;
        
        // Display results
        displayResults(taskStatus.results);
        
        // Store calibration if available
        if (taskStatus.calibration_data) {
          state.calibrationData = taskStatus.calibration_data;
        }
        
        completed = true;
        hideProgress();
        exportBtn.disabled = false;
      } else if (taskStatus.status === 'error') {
        showError('Processing error: ' + taskStatus.error);
        hideProgress();
        completed = true;
      } else {
        // Wait a bit before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  } catch (error) {
    hideProgress();
    showError('Failed to check task status: ' + error.message);
  }
}

function displayResults(results) {
  if (!results) {
    resultsContent.innerHTML = '<p class="placeholder">No results available.</p>';
    return;
  }
  
  const { detections, measurements, time_estimate } = results;
  
  // Build the HTML for the results
  let html = '';
  
  // Time estimate section
  html += `
    <div class="result-section">
      <h3>Time Estimate</h3>
      <div class="time-estimate complexity-${time_estimate.complexity}">
        ${time_estimate.estimated_minutes} minutes
        <div style="font-size: 0.8rem; font-weight: normal;">
          Complexity: ${time_estimate.complexity}
        </div>
      </div>
    </div>
  `;
  
  // Detection stats
  html += `
    <div class="result-section">
      <h3>Detected Items</h3>
  `;
  
  // Group by category
  const categories = {};
  detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  // Add category counts
  for (const [category, count] of Object.entries(categories)) {
    html += `
      <div class="result-item">
        <span>${category}:</span>
        <span>${count}</span>
      </div>
    `;
  }
  
  html += `</div>`;
  
  // Measurements section
  if (measurements && measurements.dimensions) {
    html += `
      <div class="result-section">
        <h3>Framing Measurements</h3>
    `;
    
    const { dimensions } = measurements;
    
    for (const [category, data] of Object.entries(dimensions)) {
      if (data.member_count > 0) {
        html += `
          <div class="result-item">
            <span>${category}:</span>
            <span>${data.most_common_size}</span>
          </div>
        `;
      }
    }
    
    html += `</div>`;
  }
  
  // Detailed detections
  html += `
    <div class="result-section">
      <h3>Detection Details</h3>
      <div style="max-height: 200px; overflow-y: auto;">
  `;
  
  detections.forEach((det, index) => {
    html += `
      <div class="detection-item">
        <div><strong>Item ${index + 1}:</strong> ${det.category_name}</div>
        <div>Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
      </div>
    `;
  });
  
  html += `
      </div>
    </div>
  `;
  
  // Display the results
  resultsContent.innerHTML = html;
  
  // If measurement visualization is enabled, redraw with measurements
  if (state.settings.showMeasurements) {
    visualizeMeasurements(detections, measurements);
  } else {
    visualizeDetections(detections);
  }
}

function visualizeDetections(detections) {
  if (!state.imageElement || !detections) return;
  
  // Redraw the original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Define colors for different categories
  const colors = {
    'stud': '#00FF00', // Green
    'joist': '#FF0000', // Red
    'rafter': '#0000FF', // Blue
    'beam': '#00FFFF', // Cyan
    'plate': '#FF00FF', // Magenta
    'header': '#FFFF00', // Yellow
    'electrical_box': '#800080', // Purple
    'default': '#FF8C00' // Orange (default)
  };
  
  // Draw bounding boxes for each detection
  detections.forEach(det => {
    const [x, y, w, h] = det.bbox;
    const color = colors[det.category_name] || colors.default;
    
    // Draw rectangle
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    
    // Draw label
    ctx.fillStyle = color;
    const label = `${det.category_name} (${(det.confidence * 100).toFixed(0)}%)`;
    ctx.font = '14px Arial';
    
    // Draw background for text
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x, y - 20, textWidth + 10, 20);
    
    // Draw text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, x + 5, y - 5);
  });
}

function visualizeMeasurements(detections, measurements) {
  // First draw the detections
  visualizeDetections(detections);
  
  if (!measurements || !measurements.dimensions) return;
  
  // Add measurement visualizations
  const { dimensions, spacing } = measurements;
  
  // Draw dimensions for studs, joists, etc.
  for (const [category, data] of Object.entries(dimensions)) {
    if (!data.dimensions) continue;
    
    data.dimensions.forEach(dim => {
      const [x, y, w, h] = dim.bbox;
      
      // Draw dimension labels
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px Arial';
      
      // Format the dimensions
      const formattedSize = dim.standard_size !== 'custom' 
        ? dim.standard_size
        : `${dim.thickness.toFixed(1)}x${dim.depth.toFixed(1)}`;
      
      const label = `${formattedSize} ${measurements.unit}`;
      
      // Draw at the center of the bounding box
      const centerX = x + w/2;
      const centerY = y + h/2;
      
      // Add background for better visibility
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(centerX - textWidth/2 - 5, centerY - 8, textWidth + 10, 20);
      
      // Draw text
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.fillText(label, centerX, centerY + 5);
      ctx.textAlign = 'left'; // Reset alignment
    });
  }
  
  // Draw spacing measurements if available
  if (spacing && Object.keys(spacing.spacings || {}).length > 0) {
    for (const [category, data] of Object.entries(spacing.spacings)) {
      if (!data.center_points || data.center_points.length < 2) continue;
      
      // Get the center points and draw connections
      const points = data.center_points;
      const orientation = data.orientation;
      
      // Use different colors for spacing
      ctx.strokeStyle = category === 'stud' ? '#FFA500' : '#00CED1';
      ctx.setLineDash([5, 5]); // Dashed line for spacing
      
      for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i+1];
        
        // Draw line between centers
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.stroke();
        
        // Calculate midpoint
        const midX = (p1[0] + p2[0]) / 2;
        const midY = (p1[1] + p2[1]) / 2;
        
        // Get spacing value if available
        if (data.spacings && i < data.spacings.length) {
          const spacing = data.spacings[i].toFixed(1);
          const label = `${spacing} ${measurements.unit}`;
          
          // Background for spacing label
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
          
          if (orientation === 'vertical') {
            ctx.fillRect(midX - textWidth/2 - 5, midY - 8, textWidth + 10, 20);
          } else {
            ctx.fillRect(midX - textWidth/2 - 5, midY - 8, textWidth + 10, 20);
          }
          
          // Draw spacing label
          ctx.fillStyle = '#FFFFFF';
          ctx.textAlign = 'center';
          ctx.fillText(label, midX, midY + 5);
          ctx.textAlign = 'left'; // Reset alignment
        }
      }
      
      ctx.setLineDash([]); // Reset line style
    }
  }
}

// Calibration functions
function selectCalibrationMethod(event) {
  const method = event.target.dataset.method;
  
  // Hide all methods first
  document.querySelectorAll('.calibration-method').forEach(el => {
    el.style.display = 'none';
  });
  
  // Show selected method
  if (method === 'known-object') {
    document.getElementById('calibrateKnownObject').style.display = 'block';
  } else if (method === 'reference-points') {
    document.getElementById('calibrateReferencePoints').style.display = 'block';
    
    // Reset reference points
    state.referencePoints = [];
    document.getElementById('submitPointsBtn').disabled = true;
    
    // Ensure image is displayed
    if (state.imageElement) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(state.imageElement, 0, 0);
    }
  }
}

function handleObjectTypeChange() {
  const objectType = document.getElementById('objectType').value;
  const customDimensions = document.getElementById('customDimensions');
  
  if (objectType === 'custom') {
    customDimensions.style.display = 'block';
  } else {
    customDimensions.style.display = 'none';
  }
}

function startBoxDrawing() {
  // Reset box drawing state
  state.drawingBox = true;
  state.boxStart = null;
  
  // Redraw the original image
  if (state.imageElement) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.imageElement, 0, 0);
  }
  
  // Update button
  document.getElementById('drawBoxBtn').textContent = 'Drawing...';
}

function handleCanvasMouseDown(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  if (state.drawingBox) {
    // Start drawing box
    state.boxStart = [canvasX, canvasY];
  } else if (document.getElementById('calibrateReferencePoints').style.display === 'block') {
    // Add reference point (max 2)
    if (state.referencePoints.length < 2) {
      state.referencePoints.push([canvasX, canvasY]);
      
      // Draw the point
      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw line if we have 2 points
      if (state.referencePoints.length === 2) {
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(state.referencePoints[0][0], state.referencePoints[0][1]);
        ctx.lineTo(state.referencePoints[1][0], state.referencePoints[1][1]);
        ctx.stroke();
        
        // Enable submit button
        document.getElementById('submitPointsBtn').disabled = false;
      }
    }
  } else if (state.isDrawingPath) {
    // Add path point
    state.pathPoints.push([canvasX, canvasY]);
    
    // Redraw path
    drawPath();
    
    // Enable calculate button if we have at least 2 points
    document.getElementById('calculatePathBtn').disabled = state.pathPoints.length < 2;
  }
}

function handleCanvasMouseMove(event) {
  if (!state.drawingBox || !state.boxStart) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  // Redraw the image and the current box
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw the box
  const width = canvasX - state.boxStart[0];
  const height = canvasY - state.boxStart[1];
  
  ctx.strokeStyle = '#FF0000';
  ctx.lineWidth = 2;
  ctx.strokeRect(state.boxStart[0], state.boxStart[1], width, height);
}

function handleCanvasMouseUp(event) {
  if (!state.drawingBox || !state.boxStart) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  // Scale coordinates to canvas size
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;
  
  // Complete the box
  const width = canvasX - state.boxStart[0];
  const height = canvasY - state.boxStart[1];
  
  // Store the final box (ensure positive dimensions)
  const boxX = width >= 0 ? state.boxStart[0] : canvasX;
  const boxY = height >= 0 ? state.boxStart[1] : canvasY;
  const boxWidth = Math.abs(width);
  const boxHeight = Math.abs(height);
  
  // Store the box
  state.calibrationBox = [boxX, boxY, boxWidth, boxHeight];
  
  // Reset state
  state.drawingBox = false;
  state.boxStart = null;
  
  // Update button
  document.getElementById('drawBoxBtn').textContent = 'Draw Box';
  document.getElementById('submitKnownObjectBtn').disabled = false;
}

async function submitKnownObjectCalibration() {
  if (!state.calibrationBox || !state.currentImage) {
    showError('Please draw a box around a framing member');
    return;
  }
  
  try {
    // Get object dimensions
    const objectType = document.getElementById('objectType').value;
    let dimensions;
    
    if (objectType === 'custom') {
      const width = parseFloat(document.getElementById('customWidth').value);
      const height = parseFloat(document.getElementById('customHeight').value);
      dimensions = [width, height];
    } else {
      // Predefined dimensions
      switch (objectType) {
        case '2x4':
          dimensions = [1.5, 3.5];
          break;
        case '2x6':
          dimensions = [1.5, 5.5];
          break;
        case '2x8':
          dimensions = [1.5, 7.25];
          break;
        case '4x4':
          dimensions = [3.5, 3.5];
          break;
        default:
          dimensions = [1.5, 3.5]; // Default to 2x4
      }
    }
    
    // Submit calibration request
    const result = await window.api.calibrateImage(
      state.currentImage,
      'object',
      {
        bbox: state.calibrationBox,
        dimensions: dimensions,
        units: 'inches'
      }
    );
    
    // Store calibration data
    state.calibrationData = result;
    
    // Close modal and show success message
    hideModal(calibrateModal);
    alert('Calibration successful! The system is now calibrated for accurate measurements.');
    
    // Reprocess the image with calibration
    if (state.detectionResults) {
      await processImage();
    }
  } catch (error) {
    showError('Calibration failed: ' + error.message);
  }
}

async function submitPointsCalibration() {
  if (state.referencePoints.length !== 2 || !state.currentImage) {
    showError('Please place two reference points');
    return;
  }
  
  try {
    // Get distance between points
    const distance = parseFloat(document.getElementById('referenceDistance').value);
    const unit = document.getElementById('distanceUnit').value;
    
    // Submit calibration request
    const result = await window.api.calibrateImage(
      state.currentImage,
      'points',
      {
        points: state.referencePoints,
        distance: distance,
        units: unit
      }
    );
    
    // Store calibration data
    state.calibrationData = result;
    
    // Close modal and show success message
    hideModal(calibrateModal);
    alert('Calibration successful! The system is now calibrated for accurate measurements.');
    
    // Reprocess the image with calibration
    if (state.detectionResults) {
      await processImage();
    }
  } catch (error) {
    showError('Calibration failed: ' + error.message);
  }
}

// Path drawing functions
function startPathDrawing() {
  if (!state.currentImage) {
    showError('Please select an image first');
    return;
  }
  
  // Reset path points
  state.pathPoints = [];
  state.isDrawingPath = true;
  
  // Show path drawing modal
  showModal(pathModal);
  
  // Reset path results
  document.getElementById('pathResults').style.display = 'none';
  document.getElementById('calculatePathBtn').disabled = true;
  
  // Redraw canvas
  if (state.imageElement) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.imageElement, 0, 0);
  }
}

function drawPath() {
  if (state.pathPoints.length === 0 || !state.imageElement) return;
  
  // Redraw image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw path points and lines
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 2;
  ctx.fillStyle = '#FF0000';
  
  // Draw lines between points
  if (state.pathPoints.length > 1) {
    ctx.beginPath();
    ctx.moveTo(state.pathPoints[0][0], state.pathPoints[0][1]);
    
    for (let i = 1; i < state.pathPoints.length; i++) {
      ctx.lineTo(state.pathPoints[i][0], state.pathPoints[i][1]);
    }
    
    ctx.stroke();
  }
  
  // Draw points
  state.pathPoints.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Label points
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(index + 1, point[0], point[1] - 10);
    
    // Reset for next point
    ctx.fillStyle = '#FF0000';
  });
}

function clearPath() {
  state.pathPoints = [];
  drawPath();
  document.getElementById('calculatePathBtn').disabled = true;
}

function undoLastPathPoint() {
  if (state.pathPoints.length > 0) {
    state.pathPoints.pop();
    drawPath();
    document.getElementById('calculatePathBtn').disabled = state.pathPoints.length < 2;
  }
}

async function calculatePath() {
  if (state.pathPoints.length < 2 || !state.currentImage) {
    showError('Please place at least two path points');
    return;
  }
  
  try {
    // Show progress indicator
    document.getElementById('calculatePathBtn').textContent = 'Calculating...';
    document.getElementById('calculatePathBtn').disabled = true;
    
    // Submit path calculation request
    const result = await window.api.estimatePath(
      state.currentImage,
      state.pathPoints,
      state.calibrationData
    );
    
    // Display path results
    displayPathResults(result);
    
    // Reset button
    document.getElementById('calculatePathBtn').textContent = 'Calculate Path';
    document.getElementById('calculatePathBtn').disabled = false;
  } catch (error) {
    showError('Path calculation failed: ' + error.message);
    document.getElementById('calculatePathBtn').textContent = 'Calculate Path';
    document.getElementById('calculatePathBtn').disabled = false;
  }
}

function displayPathResults(pathResult) {
  if (!pathResult) return;
  
  const resultsDiv = document.getElementById('pathResults');
  const contentDiv = document.getElementById('pathResultsContent');
  
  // Build results HTML
  let html = `
    <div class="path-result-item">
      <strong>Total Distance:</strong> ${pathResult.display_distance} ${pathResult.display_unit}
    </div>
    <div class="path-result-item">
      <strong>Drill Points:</strong> ${pathResult.drill_count}
    </div>
  `;
  
  // Add segment details
  if (pathResult.path_segments && pathResult.path_segments.length > 0) {
    html += `<h4 style="margin-top: 1rem;">Path Segments:</h4>`;
    
    pathResult.path_segments.forEach((segment, index) => {
      html += `
        <div class="path-result-item">
          <span>Segment ${index + 1}:</span>
          <span>${segment.real_distance.toFixed(2)} ${segment.unit}</span>
        </div>
      `;
    });
  }
  
  // Display drill points if available
  if (pathResult.drill_points && pathResult.drill_points.length > 0) {
    html += `<h4 style="margin-top: 1rem;">Drilling Required:</h4>`;
    
    let totalDrillTime = 0;
    const drillTimes = {
      'easy': 3,
      'moderate': 5,
      'difficult': 8
    };
    
    pathResult.drill_points.forEach((point, index) => {
      const difficulty = point.difficulty || 'moderate';
      const drillTime = drillTimes[difficulty];
      totalDrillTime += drillTime;
      
      html += `
        <div class="path-result-item">
          <span>Drill ${index + 1}:</span>
          <span>${difficulty} (approx. ${drillTime} min)</span>
        </div>
      `;
    });
    
    html += `
      <div class="path-result-item" style="margin-top: 0.5rem;">
        <strong>Estimated Drilling Time:</strong> ${totalDrillTime} minutes
      </div>
    `;
  }
  
  // Display results
  contentDiv.innerHTML = html;
  resultsDiv.style.display = 'block';
  
  // Visualize path on canvas
  visualizePath(pathResult);
}

function visualizePath(pathResult) {
  if (!state.imageElement) return;
  
  // Redraw the original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Draw path
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 3;
  
  if (pathResult.path_segments && pathResult.path_segments.length > 0) {
    pathResult.path_segments.forEach(segment => {
      const start = segment.start;
      const end = segment.end;
      
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
      
      // Draw distance label
      const midX = (start[0] + end[0]) / 2;
      const midY = (start[1] + end[1]) / 2;
      
      const label = `${segment.real_distance.toFixed(1)} ${segment.unit}`;
      
      // Add background for visibility
      ctx.font = '14px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(midX - textWidth/2 - 5, midY - 10, textWidth + 10, 20);
      
      // Draw text
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.fillText(label, midX, midY + 5);
    });
  }
  
  // Draw drill points
  if (pathResult.drill_points && pathResult.drill_points.length > 0) {
    pathResult.drill_points.forEach(point => {
      const position = point.position;
      const difficulty = point.difficulty || 'moderate';
      
      // Colors based on difficulty
      let color;
      switch (difficulty) {
        case 'easy': color = '#00FF00'; break; // Green
        case 'moderate': color = '#FFA500'; break; // Orange
        case 'difficult': color = '#FF0000'; break; // Red
        default: color = '#FFA500'; // Default orange
      }
      
      // Draw drill point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(position[0], position[1], 7, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw outer ring
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(position[0], position[1], 12, 0, 2 * Math.PI);
      ctx.stroke();
    });
  }
  
  ctx.textAlign = 'left'; // Reset alignment
}

// Settings functions
function updateConfidenceLabel() {
  const value = document.getElementById('confidenceThreshold').value;
  document.getElementById('confidenceValue').textContent = value;
}

async function saveSettings() {
  const settings = {
    apiBaseUrl: document.getElementById('apiUrl').value,
    confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value),
    autoCalibrate: document.getElementById('autoCalibrate').checked,
    showMeasurements: document.getElementById('showMeasurements').checked
  };
  
  try {
    // Save settings through API
    const updatedConfig = await window.api.saveConfig(settings);
    
    // Update local state
    state.settings = updatedConfig.settings;
    
    // Close modal
    hideModal(settingsModal);
    
    // Show success
    alert('Settings saved successfully');
    
    // If we have results, redraw them with new settings
    if (state.detectionResults) {
      displayResults(state.detectionResults);
    }
  } catch (error) {
    showError('Failed to save settings: ' + error.message);
  }
}

// Report export
function exportReport() {
  if (!state.detectionResults) {
    showError('No results to export');
    return;
  }
  
  // Create report content
  const results = state.detectionResults;
  let report = 'Electrician Time Estimate Report\n';
  report += '==============================\n\n';
  
  report += `Estimated Time: ${results.time_estimate.estimated_minutes} minutes\n`;
  report += `Complexity: ${results.time_estimate.complexity}\n\n`;
  
  // Detection counts
  report += 'Detected Items:\n';
  const categories = {};
  results.detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  for (const [category, count] of Object.entries(categories)) {
    report += `${category}: ${count}\n`;
  }
  report += '\n';
  
  // Measurements
  if (results.measurements && results.measurements.dimensions) {
    report += 'Framing Measurements:\n';
    const { dimensions } = results.measurements;
    
    for (const [category, data] of Object.entries(dimensions)) {
      if (data.member_count > 0) {
        report += `${category}: ${data.most_common_size}\n`;
      }
    }
    report += '\n';
  }
  
  // Save the report
  try {
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'electrician_report.txt';
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  } catch (error) {
    showError('Failed to export report: ' + error.message);
  }
}

// Utility functions
function showModal(modal) {
  modal.style.display = 'block';
}

function hideModal(modal) {
  modal.style.display = 'none';
}

function showProgress(percent, message) {
  progressContainer.style.display = 'block';
  progressFill.style.width = `${percent}%`;
  progressText.textContent = message || 'Processing...';
}

function hideProgress() {
  progressContainer.style.display = 'none';
}

function showError(message) {
  alert('Error: ' + message);
  console.error(message);
}

// Initialize the app when loaded
document.addEventListener('DOMContentLoaded', init);


Now, let's add a package.json file for the Electron app:

json
{
  "name": "electrician-estimator",
  "version": "1.0.0",
  "description": "AI-assisted electrician time estimation application",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "dev": "electron . --dev",
    "build": "electron-builder"
  },
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "electron": "^25.3.1",
    "electron-builder": "^24.4.0"
  },
  "dependencies": {
    "axios": "^1.4.0",
    "form-data": "^4.0.0"
  },
  "build": {
    "appId": "com.electrician.estimator",
    "productName": "Electrician Estimator",
    "files": [
      "main.js",
      "preload.js",
      "src/**/*"
    ],
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": "nsis"
    },
    "mac": {
      "target": "dmg"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}


#Now, let's create a `run.py` file that will orchestrate everything:


# run.py
import os
import argparse
import subprocess
import sys
import signal
import threading
import time
import uvicorn
from pathlib import Path

from config.settings import API_HOST, API_PORT, API_DEBUG
from utils.logger import logger

def run_api():
    """
    Start the FastAPI server
    """
    logger.info(f"Starting Electrician Time Estimator API at {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG
    )

def run_frontend():
    """
    Start the Electron frontend
    """
    logger.info("Starting Electron frontend")
    frontend_path = Path(__file__).parent / "frontend" / "electron"
    
    # Check if we need to install dependencies
    if not (frontend_path / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
    
    # Run the Electron app
    process = subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def main():
    parser = argparse.ArgumentParser(description="Run the Electrician Time Estimator application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend")
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        if 'frontend_process' in locals():
            frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.api_only:
        # Run API only
        run_api()
    elif args.frontend_only:
        # Run frontend only
        frontend_process = run_frontend()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            frontend_process.terminate()
    else:
        # Run both
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Give the API a moment to start
        time.sleep(2)
        
        # Start the frontend
        frontend_process = run_frontend()
        
        try:
            # Monitor the frontend process
            while frontend_process.poll() is None:
                time.sleep(1)
                
            # If we get here, the frontend has ended
            logger.info("Frontend process ended")
            sys.exit(0)
        except KeyboardInterrupt:
            frontend_process.terminate()
            logger.info("Application terminated by user")

if __name__ == "__main__":
    main()


Let's also create a models directory file to ensure the model weights directory is created:


# models/__init__.py
"""
Machine learning models for the electrician time estimation application.
"""

from pathlib import Path

# Create model weights directory if it doesn't exist
weights_dir = Path(__file__).parent / "weights"
weights_dir.mkdir(exist_ok=True)

# Create empty __init__.py files in subdirectories if needed
detection_dir = Path(__file__).parent / "detection"
if detection_dir.exists() and not (detection_dir / "__init__.py").exists():
    with open(detection_dir / "__init__.py", "w") as f:
        f.write('"""Detection models for electrician time estimation."""\n')

# Create empty file in models/weights directory to ensure git tracks the directory
weights_gitkeep = weights_dir / ".gitkeep"
if not weights_gitkeep.exists():
    weights_gitkeep.touch()


Finally, let's create a small script to create necessary directories:


# create_dirs.py
"""
Create necessary directories for the electrician time estimation application.
"""

from pathlib import Path

def create_dirs():
    """Create the required directory structure."""
    
    # Base directories
    dirs = [
        "data/storage/images",
        "data/storage/processed",
        "data/storage/annotations",
        "data/storage/images/tasks",
        "models/weights",
        "models/weights/checkpoints",
        "models/weights/exports",
        "logs",
        "frontend/electron/src",
        "frontend/electron/src/styles",
        "frontend/electron/src/scripts",
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    # Ensure .gitkeep files exist where needed
    keeps = [
        "data/storage/images/.gitkeep",
        "data/storage/processed/.gitkeep",
        "data/storage/annotations/.gitkeep",
        "models/weights/.gitkeep",
    ]
    
    for keep_file in keeps:
        path = Path(keep_file)
        if not path.exists():
            path.touch()
            print(f"Created file: {path}")

if __name__ == "__main__":
    create_dirs()


Let's create the api/models directory with schemas for API requests:


# api/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

class CalibrationData(BaseModel):
    """Calibration data for measurement scaling."""
    pixels_per_unit: float
    unit: str
    confidence: float
    method: str
    reference_data: Dict[str, Any] = {}

class EstimationTask(BaseModel):
    """Task information for background processing."""
    id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    image_path: Optional[str] = None
    calibration_data: Optional[CalibrationData] = None
    results: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    """Response for a submitted processing task."""
    task_id: str
    status: str

class DetectionResult(BaseModel):
    """Object detection result."""
    id: int
    bbox: List[float]  # [x, y, width, height]
    category_id: int
    category_name: str
    confidence: float

class Point(BaseModel):
    """A 2D point."""
    x: float
    y: float

class PathSegment(BaseModel):
    """A segment of a wiring path."""
    start: List[float]  # [x, y]
    end: List[float]  # [x, y]
    pixel_distance: float
    real_distance: float
    unit: str

class DrillPoint(BaseModel):
    """A point where drilling is required."""
    position: List[float]  # [x, y]
    requires_drilling: bool
    category: Optional[str] = None
    difficulty: Optional[str] = None

class PathResult(BaseModel):
    """Result of a wiring path analysis."""
    status: str
    path_segments: List[PathSegment]
    total_distance: float
    display_distance: float
    display_unit: str
    unit: str
    drill_points: List[DrillPoint]
    drill_count: int


Let's integrate all these components together with a centralized configuration system for the application settings:


# config/settings.py (expanded version)
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application settings file
CONFIG_FILE = BASE_DIR / "app_config.json"

# Default settings
DEFAULT_CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False
    },
    "models": {
        "detection": {
            "model_type": "yolov8",
            "model_size": "m",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45
        },
        "measurements": {
            "auto_calibrate": True,
            "pixels_per_inch": None  # Will be set during calibration
        }
    },
    "paths": {
        "data_dir": str(BASE_DIR / "data" / "storage"),
        "model_weights_dir": str(BASE_DIR / "models" / "weights"),
        "logs_dir": str(BASE_DIR / "logs")
    },
    "complexity": {
        "thresholds": {
            "simple": 1.5,
            "moderate": 2.5
        },
        "base_time_minutes": {
            "simple": 7,
            "moderate": 12,
            "complex": 17
        }
    }
}

# Load config from file if it exists, otherwise use defaults
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        config = DEFAULT_CONFIG
else:
    config = DEFAULT_CONFIG
    # Save default config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError:
        pass

# Override with environment variables if provided
# API settings
API_HOST = os.getenv("API_HOST", config["api"]["host"])
API_PORT = int(os.getenv("API_PORT", config["api"]["port"]))
API_DEBUG = os.getenv("API_DEBUG", str(config["api"]["debug"])).lower() in ("true", "1", "t")

# Update config with env values
config["api"]["host"] = API_HOST
config["api"]["port"] = API_PORT
config["api"]["debug"] = API_DEBUG

# Machine learning settings
MODEL_TYPE = os.getenv("MODEL_TYPE", config["models"]["detection"]["model_type"])
MODEL_SIZE = os.getenv("MODEL_SIZE", config["models"]["detection"]["model_size"])
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 
                                      config["models"]["detection"]["confidence_threshold"]))
DEFAULT_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 
                                config["models"]["detection"]["iou_threshold"]))

# Update config with env values
config["models"]["detection"]["model_type"] = MODEL_TYPE
config["models"]["detection"]["model_size"] = MODEL_SIZE
config["models"]["detection"]["confidence_threshold"] = DEFAULT_CONFIDENCE_THRESHOLD
config["models"]["detection"]["iou_threshold"] = DEFAULT_IOU_THRESHOLD

# Path settings
DATA_DIR = Path(os.getenv("DATA_DIR", config["paths"]["data_dir"]))
MODEL_WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR", config["paths"]["model_weights_dir"]))
LOGS_DIR = Path(os.getenv("LOGS_DIR", config["paths"]["logs_dir"]))

# Update config with path settings
config["paths"]["data_dir"] = str(DATA_DIR)
config["paths"]["model_weights_dir"] = str(MODEL_WEIGHTS_DIR)
config["paths"]["logs_dir"] = str(LOGS_DIR)

# Derived directories
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
TASKS_DIR = IMAGES_DIR / "tasks"

# Job complexity settings from config
COMPLEXITY_THRESHOLDS = config["complexity"]["thresholds"]
BASE_TIME_MINUTES = config["complexity"]["base_time_minutes"]

# Ensure necessary directories exist
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Function to update and save config
def update_config(new_settings):
    """Update application configuration and save to file."""
    global config
    
    # Update config with new settings
    for section, values in new_settings.items():
        if section in config:
            if isinstance(values, dict):
                for key, value in values.items():
                    if key in config[section]:
                        config[section][key] = value
            else:
                config[section] = values
    
    # Save to file
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


Finally, let's create a simple install script that can be used to set up the application:


# install.py
"""
Installation script for the Electrician Time Estimator.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if the Python version is compatible."""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required.")
        return False
    return True

def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing Python dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install Node.js dependencies for the frontend."""
    frontend_path = Path("frontend") / "electron"
    
    if not frontend_path.exists():
        print(f"Frontend directory not found: {frontend_path}")
        return False
    
    print("Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing frontend dependencies: {e}")
        return False
    except FileNotFoundError:
        print("Node.js not found. Please install Node.js to run the frontend.")
        return False

def create_directory_structure():
    """Create the necessary directory structure."""
    print("Creating directory structure...")
    try:
        subprocess.run([sys.executable, "create_dirs.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating directories: {e}")
        return False

def download_model_weights():
    """Download pre-trained model weights (optional)."""
    print("Do you want to download pre-trained YOLOv8 weights? (y/n)")
    choice = input().lower()
    
    if choice in ('y', 'yes'):
        print("Downloading YOLOv8 weights...")
        try:
            # Download YOLOv8m.pt
            subprocess.run([
                sys.executable, 
                "-c", 
                "from ultralytics import YOLO; YOLO('yolov8m.pt')"
            ], check=True)
            print("Model weights downloaded successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model weights: {e}")
            print("You can still run the application, but you'll need to download weights later.")
            return False
    
    return True

def main():
    """Main installation function."""
    print("===== Electrician Time Estimator Installation =====")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Install frontend dependencies
    frontend_ok = install_frontend_dependencies()
    if not frontend_ok:
        print("Warning: Frontend dependencies couldn't be installed.")
        print("You can still run the API-only mode.")
    
    # Download model weights
    download_model_weights()
    
    print("\nInstallation completed!")
    print("\nTo run the application:")
    print("  - Full application: python run.py")
    print("  - API only: python run.py --api-only")
    if frontend_ok:
        print("  - Frontend only: python run.py --frontend-only")

if __name__ == "__main__":
    main()


With all these components, the end-to-end integration of the electrician time estimation application is complete. This integration connects the frontend UI to the backend API, ensures proper data flow between detection, estimation, and display components, implements error handling and recovery, adds progress indicators for long-running processes, and creates a configuration system for application settings.

The solution includes:

1. A robust backend API with FastAPI for handling image processing, framing detection, and time estimation
2. A user-friendly Electron frontend for interacting with the API
3. Background task processing for handling computation-intensive operations
4. Progress indicators to keep users informed during processing
5. Configuration management with both file and environment variable support
6. Error handling and recovery mechanisms
7. Installation and setup scripts for easy deployment

The application can be run in different modes (full, API-only, frontend-only) depending on the user's needs, and the code is structured to be maintainable and extensible for future improvements.
# End-to-End Integration for Electrician Time Estimator

I'll implement the code that integrates all components of the electrician time estimation application. This integration will connect the frontend UI to the backend API, ensure proper data flow between detection, estimation, and display components, implement error handling, add progress indicators, and create a configuration system.

## 1. API Routes Integration

First, let's create the API routes that will connect our detection and measurement modules:


# api/routes/estimation.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import tempfile
import json
import os
import shutil
from typing import Dict, List, Optional, Any

from models.detection import FramingDetector, detect_framing
from models.measurements import MeasurementEstimator
from utils.logger import get_logger
from utils.exceptions import EstimatorBaseException
from config.settings import (
    MODEL_WEIGHTS_DIR, IMAGES_DIR, DEFAULT_CONFIDENCE_THRESHOLD,
    COMPLEXITY_THRESHOLDS, BASE_TIME_MINUTES
)

logger = get_logger("api_estimation")
router = APIRouter()

# Task storage for background tasks
TASKS = {}

# Initialize models lazily
detector = None
estimator = None

def load_models():
    """
    Lazy-load models when needed.
    """
    global detector, estimator
    
    if detector is None:
        # Find best available model weights
        model_paths = list(MODEL_WEIGHTS_DIR.glob("framing_detector_*.pt"))
        if model_paths:
            model_path = sorted(model_paths)[-1]  # Use most recent model
            logger.info(f"Loading detection model from {model_path}")
            detector = FramingDetector.from_checkpoint(model_path)
        else:
            logger.warning("No detection model weights found, using pretrained model")
            detector = FramingDetector(pretrained=True)
    
    if estimator is None:
        estimator = MeasurementEstimator()
    
    return detector, estimator

def process_image_task(task_id: str, image_path: Path):
    """
    Background task to process an image.
    """
    try:
        # Update task status
        TASKS[task_id]["status"] = "processing"
        
        # Load the models
        detector, estimator = load_models()
        
        # Run detection
        detection_result = detect_framing(detector, str(image_path))
        TASKS[task_id]["progress"] = 40
        
        # If we have a stored calibration, use it
        calibration_data = TASKS[task_id].get("calibration_data")
        if calibration_data:
            estimator = MeasurementEstimator(calibration_data=calibration_data)
        else:
            # Try to auto-calibrate using a known framing member
            if detection_result["detections"]:
                # Find a likely stud or joist for calibration
                for det in detection_result["detections"]:
                    if det["category_name"] in ["stud", "joist"] and det["confidence"] > 0.8:
                        # Use standard framing dimensions for calibration
                        try:
                            estimator.calibrate_from_known_object(
                                detection_result["image"],
                                det["bbox"],
                                (1.5, 3.5)  # 2x4 nominal dimensions
                            )
                            TASKS[task_id]["calibration_data"] = estimator.reference_scale.get_calibration_data()
                            break
                        except Exception as e:
                            logger.warning(f"Auto-calibration failed: {str(e)}")
        
        TASKS[task_id]["progress"] = 60
        
        # Run measurements
        measurement_result = estimator.analyze_framing_measurements(
            detection_result, calibration_check=False
        )
        TASKS[task_id]["progress"] = 80
        
        # Estimate time required based on complexity
        time_estimate = estimate_time(detection_result, measurement_result)
        
        # Save results
        results = {
            "detections": detection_result["detections"],
            "measurements": measurement_result["measurements"],
            "time_estimate": time_estimate
        }
        
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["results"] = results
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)

def estimate_time(detection_result: Dict, measurement_result: Dict) -> Dict:
    """
    Estimate time required for electrical work based on detection and measurements.
    """
    # Count different framing members
    member_counts = {}
    for det in detection_result["detections"]:
        cat = det["category_name"]
        member_counts[cat] = member_counts.get(cat, 0) + 1
    
    # Calculate complexity factors
    stud_count = member_counts.get("stud", 0)
    joist_count = member_counts.get("joist", 0)
    obstacle_count = member_counts.get("obstacle", 0) + member_counts.get("plumbing", 0)
    electrical_box_count = member_counts.get("electrical_box", 0)
    
    # Determine spacing complexity (irregular spacing is more complex)
    spacing_complexity = 0
    spacings = measurement_result.get("measurements", {}).get("spacing", {}).get("spacings", {})
    
    for cat, spacing_data in spacings.items():
        if not spacing_data.get("is_standard", True):
            spacing_complexity += 0.3
    
    # Calculate overall complexity score
    complexity_score = (
        stud_count * 0.05 +
        joist_count * 0.05 +
        obstacle_count * 0.2 +
        electrical_box_count * 0.1 +
        spacing_complexity
    )
    
    # Map to complexity level
    if complexity_score <= COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"
    elif complexity_score <= COMPLEXITY_THRESHOLDS["moderate"]:
        complexity = "moderate"
    else:
        complexity = "complex"
    
    # Calculate time estimate
    base_time = BASE_TIME_MINUTES[complexity]
    total_time = base_time + (stud_count * 0.5) + (obstacle_count * 1.5) + (electrical_box_count * 2)
    
    return {
        "complexity": complexity,
        "complexity_score": complexity_score,
        "estimated_minutes": round(total_time),
        "factors": {
            "stud_count": stud_count,
            "joist_count": joist_count,
            "obstacle_count": obstacle_count,
            "electrical_box_count": electrical_box_count,
            "spacing_complexity": spacing_complexity
        }
    }

@router.post("/estimate")
async def estimate_from_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    calibration: Optional[str] = Form(None)
):
    """
    Submit an image for time estimation (processes in background).
    Returns a task ID for checking status.
    """
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Create temp directory for this task
    task_dir = IMAGES_DIR / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    image_path = task_dir / f"input{Path(file.filename).suffix}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Parse calibration data if provided
    calibration_data = None
    if calibration:
        try:
            calibration_data = json.loads(calibration)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid calibration data format")
    
    # Create task entry
    TASKS[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "image_path": str(image_path),
        "calibration_data": calibration_data
    }
    
    # Start background processing
    background_tasks.add_task(process_image_task, task_id, image_path)
    
    return {"task_id": task_id, "status": "pending"}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a submitted task.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id].copy()
    
    # Don't return the image path in the response
    if "image_path" in task:
        del task["image_path"]
    
    return task

@router.post("/calibrate")
async def calibrate_from_image(
    file: UploadFile = File(...),
    reference_type: str = Form(...),
    reference_data: str = Form(...)
):
    """
    Calibrate the measurement system using a reference object or points.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Load the image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Initialize the estimator
        _, estimator = load_models()
        
        # Parse reference data
        ref_data = json.loads(reference_data)
        
        # Perform calibration based on reference type
        if reference_type == "points":
            if "points" not in ref_data or "distance" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing points or distance")
            
            points = ref_data["points"]
            distance = float(ref_data["distance"])
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_reference(
                image, points, distance, units
            )
        
        elif reference_type == "object":
            if "bbox" not in ref_data or "dimensions" not in ref_data:
                raise HTTPException(status_code=400, detail="Missing bbox or dimensions")
            
            bbox = ref_data["bbox"]
            dimensions = ref_data["dimensions"]
            units = ref_data.get("units", "inches")
            
            calibration = estimator.calibrate_from_known_object(
                image, bbox, dimensions, units
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid reference type")
        
        # Clean up
        os.unlink(temp_path)
        
        return calibration
        
    except EstimatorBaseException as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        logger.error(f"Calibration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during calibration")


## 2. Frontend Integration with Electron

Now let's create the Electron app that will connect to our API:

javascript
// frontend/electron/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

// Keep a global reference of the window object
let mainWindow;

// Configuration
const config = {
  apiBaseUrl: 'http://localhost:8000/api',
  settings: {
    autoCalibrate: true,
    confidenceThreshold: 0.25,
    showMeasurements: true
  }
};

// Load config from file if exists
try {
  const userDataPath = app.getPath('userData');
  const configPath = path.join(userDataPath, 'config.json');
  if (fs.existsSync(configPath)) {
    const savedConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    config.settings = {...config.settings, ...savedConfig.settings};
    if (savedConfig.apiBaseUrl) {
      config.apiBaseUrl = savedConfig.apiBaseUrl;
    }
  }
} catch (error) {
  console.error('Error loading config:', error);
}

// Save config to file
function saveConfig() {
  try {
    const userDataPath = app.getPath('userData');
    const configPath = path.join(userDataPath, 'config.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  } catch (error) {
    console.error('Error saving config:', error);
  }
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Load the index.html of the app
  mainWindow.loadFile(path.join(__dirname, 'src/index.html'));

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed
app.on('window-all-closed', function() {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function() {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// IPC handlers for communication with renderer process
ipcMain.handle('select-image', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }]
  });
  
  if (canceled) return null;
  return filePaths[0];
});

ipcMain.handle('get-config', () => {
  return config;
});

ipcMain.handle('save-config', (event, newSettings) => {
  config.settings = {...config.settings, ...newSettings};
  config.apiBaseUrl = newSettings.apiBaseUrl || config.apiBaseUrl;
  saveConfig();
  return config;
});

ipcMain.handle('submit-image', async (event, imagePath, calibrationData) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    
    if (calibrationData) {
      formData.append('calibration', JSON.stringify(calibrationData));
    }
    
    const response = await axios.post(
      `${config.apiBaseUrl}/estimate`, 
      formData,
      { 
        headers: formData.getHeaders(),
        timeout: 30000 // 30 seconds timeout
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error submitting image:', error);
    throw new Error(error.response?.data?.detail || 'Failed to submit image');
  }
});

ipcMain.handle('check-task', async (event, taskId) => {
  try {
    const response = await axios.get(
      `${config.apiBaseUrl}/tasks/${taskId}`,
      { timeout: 10000 }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error checking task:', error);
    throw new Error(error.response?.data?.detail || 'Failed to check task status');
  }
});


## 3. Preload Script to Bridge Electron Main and Renderer Processes

javascript
// frontend/electron/preload.js
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'api', {
    selectImage: () => ipcRenderer.invoke('select-image'),
    getConfig: () => ipcRenderer.invoke('get-config'),
    saveConfig: (settings) => ipcRenderer.invoke('save-config', settings),
    submitImage: (imagePath, calibrationData) => 
      ipcRenderer.invoke('submit-image', imagePath, calibrationData),
    checkTask: (taskId) => ipcRenderer.invoke('check-task', taskId),
    calibrateImage: (imagePath, referenceType, referenceData) => 
      ipcRenderer.invoke('calibrate-image', imagePath, referenceType, referenceData),
    estimatePath: (imagePath, pathPoints, calibrationData) => 
      ipcRenderer.invoke('estimate-path', imagePath, pathPoints, calibrationData)
  }
);


## 4. Frontend UI Implementation

html
<!-- frontend/electron/src/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electrician Time Estimator</title>
    <link rel="stylesheet" href="styles/main.css">
</head>
<body>
    <header>
        <h1>Electrician Time Estimator</h1>
        <div class="toolbar">
            <button id="selectImageBtn">Select Image</button>
            <button id="calibrateBtn">Calibrate</button>
            <button id="settingsBtn">Settings</button>
        </div>
    </header>

    <main>
        <div id="imageContainer">
            <div id="placeholder">
                <p>Select an image to analyze</p>
            </div>
            <div id="canvasContainer" style="display: none;">
                <canvas id="mainCanvas"></canvas>
            </div>
            <div id="progress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <p class="progress-text">Processing...</p>
            </div>
        </div>

        <div id="resultsPanel">
            <div class="panel-header">
                <h2>Results</h2>
                <button id="exportBtn" disabled>Export Report</button>
            </div>
            <div id="resultsContent">
                <p class="placeholder">No results yet. Process an image to see time estimates.</p>
            </div>
        </div>
    </main>

    <!-- Settings Modal -->
    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2>Settings</h2>
            
            <div class="settings-form">
                <div class="form-group">
                    <label for="apiUrl">API URL:</label>
                    <input type="text" id="apiUrl" placeholder="http://localhost:8000/api">
                </div>
                
                <div class="form-group">
                    <label for="confidenceThreshold">Detection Confidence Threshold:</label>
                    <input type="range" id="confidenceThreshold" min="0.05" max="0.95" step="0.05" value="0.25">
                    <span id="confidenceValue">0.25</span>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="autoCalibrate" checked>
                    <label for="autoCalibrate">Auto-calibrate using detected framing members</label>
                </div>
                
                <div class="form-group checkbox">
                    <input type="checkbox" id="showMeasurements" checked>
                    <label for="showMeasurements">Show measurements on detection</label>
                </div>
                
                <button id="saveSettingsBtn">Save Settings</button>
            </div>
        </div>
    </div>

    <script src="scripts/main.js"></script>
</body>
</html>


## 5. Main Frontend JavaScript with Error Handling

javascript
// frontend/electron/src/scripts/main.js
// Global state
let state = {
  currentImage: null,
  imageElement: null,
  calibrationData: null,
  taskId: null,
  detectionResults: null,
  settings: {
    apiBaseUrl: 'http://localhost:8000/api',
    confidenceThreshold: 0.25,
    autoCalibrate: true,
    showMeasurements: true
  }
};

// Canvas for image display and interaction
let canvas = document.getElementById('mainCanvas');
let ctx = canvas.getContext('2d');

// UI Elements
const selectImageBtn = document.getElementById('selectImageBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const settingsBtn = document.getElementById('settingsBtn');
const exportBtn = document.getElementById('exportBtn');
const placeholder = document.getElementById('placeholder');
const canvasContainer = document.getElementById('canvasContainer');
const progressContainer = document.getElementById('progress');
const progressFill = document.querySelector('.progress-fill');
const progressText = document.querySelector('.progress-text');
const resultsContent = document.getElementById('resultsContent');
const settingsModal = document.getElementById('settingsModal');
const closeButtons = document.querySelectorAll('.close-btn');

// Initialize the app
async function init() {
  // Load settings
  try {
    const config = await window.api.getConfig();
    state.settings = {...state.settings, ...config.settings};
    if (config.apiBaseUrl) {
      state.settings.apiBaseUrl = config.apiBaseUrl;
    }
    
    // Update UI with settings
    document.getElementById('apiUrl').value = state.settings.apiBaseUrl;
    document.getElementById('confidenceThreshold').value = state.settings.confidenceThreshold;
    document.getElementById('confidenceValue').textContent = state.settings.confidenceThreshold;
    document.getElementById('autoCalibrate').checked = state.settings.autoCalibrate;
    document.getElementById('showMeasurements').checked = state.settings.showMeasurements;
  } catch (error) {
    console.error('Error loading settings:', error);
    showError('Failed to load settings: ' + error.message);
  }
  
  // Set up event listeners
  setupEventListeners();
}

function setupEventListeners() {
  // Main buttons
  selectImageBtn.addEventListener('click', handleSelectImage);
  calibrateBtn.addEventListener('click', () => showModal(settingsModal));
  settingsBtn.addEventListener('click', () => showModal(settingsModal));
  exportBtn.addEventListener('click', exportReport);
  
  // Close modal buttons
  closeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const modal = btn.closest('.modal');
      hideModal(modal);
    });
  });
  
  // Settings modal
  document.getElementById('confidenceThreshold').addEventListener('input', updateConfidenceLabel);
  document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
}

// Image selection and processing
async function handleSelectImage() {
  try {
    const imagePath = await window.api.selectImage();
    if (!imagePath) return;
    
    // Reset state
    state.currentImage = imagePath;
    state.detectionResults = null;
    state.taskId = null;
    exportBtn.disabled = true;
    
    // Show the image in canvas
    loadImageToCanvas(imagePath);
    
    // Hide placeholder, show canvas
    placeholder.style.display = 'none';
    canvasContainer.style.display = 'block';
    
    // Process the image
    await processImage();
  } catch (error) {
    showError('Failed to select image: ' + error.message);
  }
}

function loadImageToCanvas(imagePath) {
  // Create an image element
  const img = new Image();
  img.onload = function() {
    // Set canvas dimensions to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw image to canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    // Store image element
    state.imageElement = img;
  };
  
  // Set image source to file path (using URL scheme for Electron)
  img.src = 'file://' + imagePath;
}

async function processImage() {
  try {
    // Show progress bar
    showProgress(10, 'Submitting image...');
    
    // Submit the image for processing
    const taskResponse = await window.api.submitImage(
      state.currentImage, 
      state.calibrationData
    );
    
    state.taskId = taskResponse.task_id;
    
    // Poll for task completion
    await pollTaskStatus();
  } catch (error) {
    hideProgress();
    showError('Failed to process image: ' + error.message);
  }
}

async function pollTaskStatus() {
  if (!state.taskId) return;
  
  try {
    let completed = false;
    while (!completed) {
      const taskStatus = await window.api.checkTask(state.taskId);
      
      // Update progress
      showProgress(taskStatus.progress || 10, `${taskStatus.status}...`);
      
      if (taskStatus.status === 'completed') {
        // Process is complete
        state.detectionResults = taskStatus.results;
        
        // Display results
        displayResults(taskStatus.results);
        
        // Store calibration if available
        if (taskStatus.calibration_data) {
          state.calibrationData = taskStatus.calibration_data;
        }
        
        completed = true;
        hideProgress();
        exportBtn.disabled = false;
      } else if (taskStatus.status === 'error') {
        showError('Processing error: ' + taskStatus.error);
        hideProgress();
        completed = true;
      } else {
        // Wait a bit before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  } catch (error) {
    hideProgress();
    showError('Failed to check task status: ' + error.message);
  }
}

function displayResults(results) {
  if (!results) {
    resultsContent.innerHTML = '<p class="placeholder">No results available.</p>';
    return;
  }
  
  const { detections, measurements, time_estimate } = results;
  
  // Build the HTML for the results
  let html = '';
  
  // Time estimate section
  html += `
    <div class="result-section">
      <h3>Time Estimate</h3>
      <div class="time-estimate complexity-${time_estimate.complexity}">
        ${time_estimate.estimated_minutes} minutes
        <div style="font-size: 0.8rem; font-weight: normal;">
          Complexity: ${time_estimate.complexity}
        </div>
      </div>
    </div>
  `;
  
  // Detection stats
  html += `
    <div class="result-section">
      <h3>Detected Items</h3>
  `;
  
  // Group by category
  const categories = {};
  detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  // Add category counts
  for (const [category, count] of Object.entries(categories)) {
    html += `
      <div class="result-item">
        <span>${category}:</span>
        <span>${count}</span>
      </div>
    `;
  }
  
  html += `</div>`;
  
  // Measurements section
  if (measurements && measurements.dimensions) {
    html += `
      <div class="result-section">
        <h3>Framing Measurements</h3>
    `;
    
    const { dimensions } = measurements;
    
    for (const [category, data] of Object.entries(dimensions)) {
      if (data.member_count > 0) {
        html += `
          <div class="result-item">
            <span>${category}:</span>
            <span>${data.most_common_size}</span>
          </div>
        `;
      }
    }
    
    html += `</div>`;
  }
  
  // Display the results
  resultsContent.innerHTML = html;
  
  // Visualize the detections on the image
  visualizeDetections(detections);
}

function visualizeDetections(detections) {
  if (!state.imageElement || !detections) return;
  
  // Redraw the original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageElement, 0, 0);
  
  // Define colors for different categories
  const colors = {
    'stud': '#00FF00', // Green
    'joist': '#FF0000', // Red
    'rafter': '#0000FF', // Blue
    'beam': '#00FFFF', // Cyan
    'plate': '#FF00FF', // Magenta
    'header': '#FFFF00', // Yellow
    'electrical_box': '#800080', // Purple
    'default': '#FF8C00' // Orange (default)
  };
  
  // Draw bounding boxes for each detection
  detections.forEach(det => {
    const [x, y, w, h] = det.bbox;
    const color = colors[det.category_name] || colors.default;
    
    // Draw rectangle
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    
    // Draw label
    ctx.fillStyle = color;
    const label = `${det.category_name} (${(det.confidence * 100).toFixed(0)}%)`;
    ctx.font = '14px Arial';
    
    // Draw background for text
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(x, y - 20, textWidth + 10, 20);
    
    // Draw text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, x + 5, y - 5);
  });
}

// Settings functions
function updateConfidenceLabel() {
  const value = document.getElementById('confidenceThreshold').value;
  document.getElementById('confidenceValue').textContent = value;
}

async function saveSettings() {
  const settings = {
    apiBaseUrl: document.getElementById('apiUrl').value,
    confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value),
    autoCalibrate: document.getElementById('autoCalibrate').checked,
    showMeasurements: document.getElementById('showMeasurements').checked
  };
  
  try {
    // Save settings through API
    const updatedConfig = await window.api.saveConfig(settings);
    
    // Update local state
    state.settings = updatedConfig.settings;
    
    // Close modal
    hideModal(settingsModal);
    
    // Show success
    alert('Settings saved successfully');
  } catch (error) {
    showError('Failed to save settings: ' + error.message);
  }
}

// Report export
function exportReport() {
  if (!state.detectionResults) {
    showError('No results to export');
    return;
  }
  
  // Create report content
  const results = state.detectionResults;
  let report = 'Electrician Time Estimate Report\n';
  report += '==============================\n\n';
  
  report += `Estimated Time: ${results.time_estimate.estimated_minutes} minutes\n`;
  report += `Complexity: ${results.time_estimate.complexity}\n\n`;
  
  // Detection counts
  report += 'Detected Items:\n';
  const categories = {};
  results.detections.forEach(det => {
    if (!categories[det.category_name]) {
      categories[det.category_name] = 0;
    }
    categories[det.category_name]++;
  });
  
  for (const [category, count] of Object.entries(categories)) {
    report += `${category}: ${count}\n`;
  }
  
  try {
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'electrician_report.txt';
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  } catch (error) {
    showError('Failed to export report: ' + error.message);
  }
}

// Utility functions
function showModal(modal) {
  modal.style.display = 'block';
}

function hideModal(modal) {
  modal.style.display = 'none';
}

function showProgress(percent, message) {
  progressContainer.style.display = 'block';
  progressFill.style.width = `${percent}%`;
  progressText.textContent = message || 'Processing...';
}

function hideProgress() {
  progressContainer.style.display = 'none';
}

function showError(message) {
  alert('Error: ' + message);
  console.error(message);
}

// Initialize the app when loaded
document.addEventListener('DOMContentLoaded', init);


## 6. Centralized Configuration System


# config/settings.py (expanded version)
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application settings file
CONFIG_FILE = BASE_DIR / "app_config.json"

# Default settings
DEFAULT_CONFIG = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "debug": False
    },
    "models": {
        "detection": {
            "model_type": "yolov8",
            "model_size": "m",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45
        },
        "measurements": {
            "auto_calibrate": True,
            "pixels_per_inch": None
        }
    },
    "paths": {
        "data_dir": str(BASE_DIR / "data" / "storage"),
        "model_weights_dir": str(BASE_DIR / "models" / "weights"),
        "logs_dir": str(BASE_DIR / "logs")
    },
    "complexity": {
        "thresholds": {
            "simple": 1.5,
            "moderate": 2.5
        },
        "base_time_minutes": {
            "simple": 7,
            "moderate": 12,
            "complex": 17
        }
    }
}

# Load config from file if it exists, otherwise use defaults
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        config = DEFAULT_CONFIG
else:
    config = DEFAULT_CONFIG
    # Save default config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError:
        pass

# Extract main settings for use in the application
API_HOST = os.getenv("API_HOST", config["api"]["host"])
API_PORT = int(os.getenv("API_PORT", config["api"]["port"]))
API_DEBUG = os.getenv("API_DEBUG", str(config["api"]["debug"])).lower() in ("true", "1", "t")

# Machine learning settings
MODEL_TYPE = os.getenv("MODEL_TYPE", config["models"]["detection"]["model_type"])
DEFAULT_CONFIDENCE_THRESHOLD = float(config["models"]["detection"]["confidence_threshold"])

# Path settings
DATA_DIR = Path(config["paths"]["data_dir"])
MODEL_WEIGHTS_DIR = Path(config["paths"]["model_weights_dir"])
LOGS_DIR = Path(config["paths"]["logs_dir"])

# Derived directories
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Job complexity settings
COMPLEXITY_THRESHOLDS = config["complexity"]["thresholds"]
BASE_TIME_MINUTES = config["complexity"]["base_time_minutes"]

# Ensure necessary directories exist
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Function to update and save config
def update_config(new_settings):
    """Update application configuration and save to file."""
    global config
    
    # Update config with new settings
    for section, values in new_settings.items():
        if section in config:
            if isinstance(values, dict):
                for key, value in values.items():
                    if key in config[section]:
                        config[section][key] = value
            else:
                config[section] = values
    
    # Save to file
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except IOError:
        return False


## 7. Main Application Runner with Error Handling


# run.py
import os
import argparse
import subprocess
import sys
import signal
import threading
import time
import uvicorn
from pathlib import Path

from config.settings import API_HOST, API_PORT, API_DEBUG
from utils.logger import logger

def run_api():
    """
    Start the FastAPI server
    """
    logger.info(f"Starting Electrician Time Estimator API at {API_HOST}:{API_PORT}")
    try:
        uvicorn.run(
            "api.main:app",
            host=API_HOST,
            port=API_PORT,
            reload=API_DEBUG
        )
    except Exception as e:
        logger.error(f"API server error: {str(e)}")
        sys.exit(1)

def run_frontend():
    """
    Start the Electron frontend
    """
    logger.info("Starting Electron frontend")
    frontend_path = Path(__file__).parent / "frontend" / "electron"
    
    # Check if the frontend directory exists
    if not frontend_path.exists():
        logger.error(f"Frontend directory not found: {frontend_path}")
        return None
    
    # Check if we need to install dependencies
    if not (frontend_path / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install frontend dependencies: {str(e)}")
            return None
        except FileNotFoundError:
            logger.error("Node.js not found. Please install Node.js to run the frontend.")
            return None
    
    # Run the Electron app
    try:
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    except Exception as e:
        logger.error(f"Failed to start frontend: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run the Electrician Time Estimator application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend")
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        if 'frontend_process' in locals() and frontend_process:
            frontend_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.api_only:
        # Run API only
        run_api()
    elif args.frontend_only:
        # Run frontend only
        frontend_process = run_frontend()
        if not frontend_process:
            logger.error("Failed to start frontend")
            sys.exit(1)
            
        try:
            # Monitor frontend process
            exit_code = frontend_process.wait()
            logger.info(f"Frontend process exited with code {exit_code}")
        except KeyboardInterrupt:
            frontend_process.terminate()
            logger.info("Frontend terminated by user")
    else:
        # Run both API and frontend
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Give the API a moment to start
        time.sleep(2)
        
        # Start the frontend
        frontend_process = run_frontend()
        if not frontend_process:
            logger.error("Failed to start frontend")
            sys.exit(1)
        
        try:
            # Monitor the frontend process
            exit_code = frontend_process.wait()
            logger.info(f"Frontend process exited with code {exit_code}")
            sys.exit(0)
        except KeyboardInterrupt:
            if frontend_process:
                frontend_process.terminate()
            logger.info("Application terminated by user")

if __name__ == "__main__":
    main()


## 8. CSS Styles for Frontend

css
/* frontend/electron/src/styles/main.css */
:root {
    --primary-color: #3498db;
    --primary-dark: #2980b9;
    --secondary-color: #2ecc71;
    --accent-color: #e74c3c;
    --text-color: #333;
    --light-gray: #f4f4f4;
    --medium-gray: #ddd;
    --dark-gray: #888;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: var(--text-color);
    line-height: 1.6;
    background-color: #f9f9f9;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

header {
    background-color: var(--primary-color);
    color: white;
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

main {
    display: flex;
    flex: 1;
    padding: 1rem;
    gap: 1rem;
    overflow: hidden;
}

button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

button:hover {
    background-color: var(--primary-dark);
}

button:disabled {
    background-color: var(--dark-gray);
    cursor: not-allowed;
}

.toolbar {
    display: flex;
    gap: 0.5rem;
}

#imageContainer {
    flex: 1.5;
    background-color: var(--light-gray);
    border-radius: 4px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
}

#placeholder {
    text-align: center;
    color: var(--dark-gray);
}

#canvasContainer {
    width: 100%;
    height: 100%;
    position: relative;
}

canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

#resultsPanel {
    flex: 1;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.panel-header {
    padding: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--light-gray);
    border-bottom: 1px solid var(--medium-gray);
}

#resultsContent {
    padding: 1rem;
    overflow-y: auto;
    flex: 1;
}

/* Progress bar styles */
#progress {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 1rem;
    border-top: 1px solid var(--medium-gray);
}

.progress-bar {
    height: 10px;
    background-color: var(--light-gray);
    border-radius: 5px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--secondary-color);
    width: 0%;
    transition: width 0.3s ease-in-out;
}

.progress-text {
    text-align: center;
    margin-top: 0.5rem;
    font-size: 0.9rem;
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    z-index: 100;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background-color: white;
    margin: 10% auto;
    padding: 1.5rem;
    border-radius: 6px;
    width: 60%;
    max-width: 600px;
    position: relative;
}

.close-btn {
    position: absolute;
    top: 0.75rem;
    right: 1rem;
    font-size: 1.5rem;
    cursor: pointer;
}

/* Result styling */
.result-section {
    margin-bottom: 1.5rem;
}

.result-section h3 {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--light-gray);
}

.result-item {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
}

.time-estimate {
    font-size: 1.5rem;
    text-align: center;
    padding: 1rem;
    margin: 1rem 0;
    background-color: var(--light-gray);
    border-radius: 4px;
    font-weight: bold;
}

.complexity-simple { color: var(--secondary-color); }
.complexity-moderate { color: orange; }
.complexity-complex { color: var(--accent-color); }


## 9. Package.json for Electron App

json
{
  "name": "electrician-estimator-frontend",
  "version": "1.0.0",
  "description": "Frontend for electrician time estimation app",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "dev": "NODE_ENV=development electron .",
    "build": "electron-builder"
  },
  "dependencies": {
    "axios": "^1.4.0",
    "form-data": "^4.0.0"
  },
  "devDependencies": {
    "electron": "^25.3.1",
    "electron-builder": "^24.4.0"
  },
  "build": {
    "appId": "com.electrician.estimator",
    "productName": "Electrician Estimator",
    "win": {
      "target": "nsis"
    },
    "mac": {
      "target": "dmg"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}


## Application Overview

This integration provides:

1. **Data Flow Integration**: Connects the frontend UI to the backend API, ensuring proper flow between detection, measurement, and time estimation components.

2. **Error Handling**: Implements comprehensive error handling and recovery at all levels:
   - Backend exceptions are caught and returned as structured responses
   - API timeouts and connection errors are handled in the frontend
   - Image processing failures trigger appropriate user notifications

3. **Progress Indicators**: Adds a progress bar system for long-running tasks:
   - Background task processing with progress updates
   - Real-time status reporting to the frontend
   - Visual progress feedback for users

4. **Configuration System**: Creates a flexible configuration system that:
   - Loads from environment variables
   - Supports a central config file
   - Allows settings to be changed through the UI

5. **Cross-Platform Support**: Built using Electron for the frontend and FastAPI for the backend, ensuring compatibility across Windows, macOS, and Linux.

This end-to-end integration creates a complete application that can detect framing members in residential construction images, estimate measurements, and provide electricians with time estimates for completing their work.
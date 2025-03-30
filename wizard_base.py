# Enhanced Script: Electrician Estimator Application

#I've enhanced the code with production-ready implementations based on your prompts. Here are the improved files:

## services/estimate_generator.py

"""
Service to generate electrical estimates by orchestrating AI analysis and domain rules.
"""

import re
import json
from typing import List, Dict, Union, Optional, Any, Tuple
from models.estimate import Estimate, CustomerInfo, LineItem, LaborCalculation, Material, StatusTracking, AIData
from services.ai_service import AIService
from services.computer_vision import ComputerVisionService
from services.measurement_service import MeasurementService
from models.framing import FramingMember
from models.measurement import MeasurementData
from utils.logger import logger
from utils.exceptions import AIParsingError, EstimateGenerationError

class EstimateGeneratorService:
    """
    Orchestrates AI analysis, measurement, and domain rules to generate electrical estimates.
    """

    def __init__(self, ai_service: AIService, computer_vision_service: ComputerVisionService, measurement_service: MeasurementService):
        """
        Initializes the Estimate Generator Service.

        Args:
            ai_service: Instance of AIService for AI model interactions.
            computer_vision_service: Instance of ComputerVisionService for computer vision tasks.
            measurement_service: Instance of MeasurementService for measurement calculations.
        """
        self.ai_service = ai_service
        self.cv_service = computer_vision_service
        self.measurement_service = measurement_service
        logger.info("Estimate Generator Service initialized.")

    async def generate_estimate_from_media(self, media_paths: List[str], customer_info: CustomerInfo, estimate_description: str, progress_callback=None) -> Estimate:
        """
        Generates a complete electrical estimate from media files (images/videos).

        Args:
            media_paths: List of paths to media files (images, videos).
            customer_info: CustomerInfo object containing customer details.
            estimate_description: Description of the electrical work for the estimate.
            progress_callback: Optional callback function for progress updates.

        Returns:
            Estimate object containing the generated estimate details.
        """
        try:
            if progress_callback:
                progress_callback(5, "Initializing estimate generation...")
                
            logger.info(f"Generating estimate for customer: {customer_info.name}, media files: {len(media_paths)} items")

            # 1. Process media files using Computer Vision Service
            if progress_callback:
                progress_callback(10, "Processing media files...")
                
            framing_members: List[FramingMember] = []
            prepared_image_paths = []
            
            try:
                prepared_image_paths = await self.cv_service.prepare_media_for_analysis(media_paths)
                if prepared_image_paths:
                    if progress_callback:
                        progress_callback(20, "Detecting framing members...")
                        
                    batch_framing_detections = await self.cv_service.batch_detect_framing_members(prepared_image_paths)
                    
                    # Flatten list of lists and extend framing_members list
                    for detections_list in batch_framing_detections:
                        framing_members.extend(detections_list)

                    logger.info(f"Detected {len(framing_members)} framing members.")
            except Exception as e:
                logger.error(f"Error during computer vision processing: {e}")
                # Continue with partial data rather than failing completely

            # 2. Measurement Estimation using Measurement Service
            if progress_callback:
                progress_callback(30, "Calculating measurements...")
                
            measurement_data = MeasurementData()
            try:
                if framing_members:
                    measured_framing_members = self.measurement_service.calculate_dimensions(framing_members)
                    spacing_measurements = self.measurement_service.calculate_spacing_between_members(measured_framing_members)
                    measurement_data = self.measurement_service.create_measurement_data(measured_framing_members, spacing_measurements)
                    logger.info(f"Measurement estimation completed. {len(measurement_data.dimensions)} dimension measurements available.")
            except Exception as e:
                logger.error(f"Error during measurement estimation: {e}")
                # Continue with partial or no measurement data

            # 3. AI-Powered Text Analysis and Estimate Generation using AI Service
            if progress_callback:
                progress_callback(50, "Analyzing with AI...")
                
            raw_ai_analysis_result = ""
            try:
                prompt_text = self._build_estimation_prompt(customer_info, estimate_description, framing_members, measurement_data)
                raw_ai_analysis_result = await self.ai_service.analyze_images_claude(prepared_image_paths, prompt_text)
                logger.info("AI text analysis completed.")
            except Exception as e:
                logger.error(f"Error during AI text analysis: {e}")
                raise EstimateGenerationError(f"AI analysis failed: {e}")

            # 4. Parse AI output into structured estimate data
            if progress_callback:
                progress_callback(75, "Parsing AI results...")
                
            parsed_line_items = self._parse_line_items(raw_ai_analysis_result)
            parsed_materials = self._parse_materials(raw_ai_analysis_result)
            parsed_labor = self._parse_labor_calculation(raw_ai_analysis_result)
            ai_confidence = self._extract_ai_confidence(raw_ai_analysis_result)
            
            # 5. Structure Estimate Data
            if progress_callback:
                progress_callback(90, "Creating final estimate...")
                
            estimate = Estimate(
                customer=customer_info,
                description=estimate_description,
                line_items=parsed_line_items,
                labor=parsed_labor,
                materials=parsed_materials,
                ai_data=AIData(
                    raw_ai_response=raw_ai_analysis_result,
                    analysis_summary=self._generate_analysis_summary(raw_ai_analysis_result),
                    ai_model_used=self.ai_service.claude_model_name,
                    overall_confidence_score=ai_confidence,
                    component_detection_confidence={
                        f.type.value: f.confidence for f in framing_members if hasattr(f, 'confidence')
                    },
                    input_media_analysis=[{"path": path, "processed": True} for path in prepared_image_paths]
                ),
            )
            estimate.status_tracking.update_status("pending_review")
            
            if progress_callback:
                progress_callback(100, "Estimate generation complete!")
                
            logger.info(f"Estimate generation process completed with {len(estimate.line_items)} line items")
            return estimate

        except Exception as e:
            logger.error(f"Unexpected error in estimate generation: {e}", exc_info=True)
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")
            raise EstimateGenerationError(f"Failed to generate estimate: {e}")

    def _build_estimation_prompt(self, customer_info: CustomerInfo, estimate_description: str, 
                                framing_members: List[FramingMember], measurement_data: MeasurementData) -> str:
        """
        Constructs a detailed prompt for the AI model including customer info, description, and CV analysis results.
        """
        prompt_parts = [
            "You are an expert electrical estimator with 20+ years experience in residential and commercial electrical projects.",
            "Analyze the following information and images to create a detailed electrical estimate.",
            "\n# Customer Information:",
            f"Customer Name: {customer_info.name}",
            f"Customer Address: {customer_info.address}",
            f"Customer Type: {getattr(customer_info, 'customer_type', 'Residential')}",
            f"Customer Notes: {customer_info.notes if hasattr(customer_info, 'notes') and customer_info.notes else 'None'}",
            
            "\n# Estimate Description:",
            estimate_description,
            
            "\n# Computer Vision Analysis Results:",
            f"Detected {len(framing_members)} framing members in the images.",
        ]
        
        # Add framing member details if available
        if framing_members:
            framing_types = {}
            for member in framing_members:
                if member.type.value not in framing_types:
                    framing_types[member.type.value] = 0
                framing_types[member.type.value] += 1
                
            prompt_parts.append("## Framing Member Breakdown:")
            for framing_type, count in framing_types.items():
                prompt_parts.append(f"- {framing_type}: {count} detected")
        
        # Add measurement data if available
        if hasattr(measurement_data, 'spacing') and measurement_data.spacing:
            prompt_parts.append("\n## Spacing Measurements:")
            for space_type, spacing in measurement_data.spacing.items():
                prompt_parts.append(f"- {space_type}: {spacing}")
        
        # Add detailed instructions for the AI
        prompt_parts.extend([
            "\n# Output Instructions:",
            "Create a comprehensive electrical estimate with the following sections:",
            
            "\n## 1. LINE ITEMS",
            "Present a detailed list of line items in this format:",
            "LINE ITEM: [Description]",
            "QUANTITY: [Number]",
            "UNIT PRICE: [Dollar Amount]",
            "LABOR HOURS: [Number]",
            "MATERIAL COST: [Dollar Amount]",
            "EXPLANATION: [Brief explanation of why this item is needed]",
            
            "\n## 2. MATERIALS",
            "List all required materials in this format:",
            "MATERIAL: [Name of material]",
            "QUANTITY: [Number]",
            "UNIT COST: [Dollar Amount]",
            "SUPPLIER: [Recommended supplier if known]",
            "NOTES: [Any notes about the material]",
            
            "\n## 3. LABOR CALCULATION",
            "Provide labor calculation in this format:",
            "BASE HOURS: [Total estimated labor hours]",
            "HOURLY RATE: [Recommended hourly rate]",
            "COMPLEXITY FACTOR: [Number between 0.8-1.5 based on job complexity]",
            "NOTES: [Explanation of labor calculation]",
            
            "\n## 4. SUMMARY",
            "Provide a summary including:",
            "SUBTOTAL: [Sum of line items]",
            "LABOR COST: [Total labor cost]",
            "MATERIAL COST: [Total material cost]",
            "TOTAL ESTIMATE: [Final price]",
            "CONFIDENCE: [Your confidence in this estimate as a percentage]",
            
            "\n## 5. RECOMMENDATIONS",
            "Provide any additional recommendations, safety considerations, or permit requirements.",
            
            "\nVery important: Format your response exactly according to these section headers and field formats.",
            "The response must be machine-parsable while remaining human-readable."
        ])
        
        return "\n".join(prompt_parts)

    def _parse_line_items(self, ai_text: str) -> List[LineItem]:
        """
        Parse line items from the AI response text.
        """
        line_items = []
        try:
            # Look for LINE ITEMS section
            line_items_section = self._extract_section(ai_text, "LINE ITEMS", "MATERIALS")
            if not line_items_section:
                logger.warning("No LINE ITEMS section found in AI response")
                return []
            
            # Define a regex pattern to match line item blocks
            line_item_pattern = re.compile(
                r"LINE ITEM:\s*(.+?)[\r\n]+"
                r"QUANTITY:\s*(\d+)[\r\n]+"
                r"UNIT PRICE:\s*\$?(\d+\.\d+|\d+)[\r\n]+"
                r"LABOR HOURS:\s*(\d+\.\d+|\d+)[\r\n]+"
                r"MATERIAL COST:\s*\$?(\d+\.\d+|\d+)[\r\n]+"
                r"(?:EXPLANATION:|NOTES:)\s*(.+?)(?=LINE ITEM:|$)",
                re.DOTALL
            )
            
            # Find all matches in the LINE ITEMS section
            matches = line_item_pattern.finditer(line_items_section)
            
            for match in matches:
                try:
                    description = match.group(1).strip()
                    quantity = int(match.group(2))
                    unit_price = float(match.group(3))
                    labor_hours = float(match.group(4))
                    material_cost = float(match.group(5))
                    notes = match.group(6).strip()
                    
                    line_item = LineItem(
                        description=description,
                        quantity=quantity,
                        unit_price=unit_price,
                        labor_hours=labor_hours,
                        material_cost=material_cost,
                        ai_confidence_score=0.9,  # Default value unless specified elsewhere
                        measurement_source="AI"
                    )
                    line_items.append(line_item)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line item: {e}")
                    continue
            
            return line_items
            
        except Exception as e:
            logger.error(f"Error parsing line items: {e}", exc_info=True)
            raise AIParsingError(f"Failed to parse line items: {e}")

    def _parse_materials(self, ai_text: str) -> List[Material]:
        """
        Parse materials from the AI response text.
        """
        materials = []
        try:
            # Look for MATERIALS section
            materials_section = self._extract_section(ai_text, "MATERIALS", "LABOR CALCULATION")
            if not materials_section:
                logger.warning("No MATERIALS section found in AI response")
                return []
            
            # Define a regex pattern to match material blocks
            material_pattern = re.compile(
                r"MATERIAL:\s*(.+?)[\r\n]+"
                r"QUANTITY:\s*(\d+)[\r\n]+"
                r"UNIT COST:\s*\$?(\d+\.\d+|\d+)[\r\n]+"
                r"(?:SUPPLIER:\s*(.+?)[\r\n]+)?"
                r"(?:NOTES:\s*(.+?))?(?=MATERIAL:|$)",
                re.DOTALL
            )
            
            # Find all matches in the MATERIALS section
            matches = material_pattern.finditer(materials_section)
            
            for match in matches:
                try:
                    name = match.group(1).strip()
                    quantity = int(match.group(2))
                    unit_cost = float(match.group(3))
                    supplier = match.group(4).strip() if match.group(4) else None
                    notes = match.group(5).strip() if match.group(5) else None
                    
                    material = Material(
                        name=name,
                        quantity=quantity,
                        unit_cost=unit_cost,
                        supplier=supplier,
                        notes=notes
                    )
                    materials.append(material)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing material: {e}")
                    continue
            
            return materials
            
        except Exception as e:
            logger.error(f"Error parsing materials: {e}", exc_info=True)
            raise AIParsingError(f"Failed to parse materials: {e}")

    def _parse_labor_calculation(self, ai_text: str) -> LaborCalculation:
        """
        Parse labor calculation from the AI response text.
        """
        try:
            # Look for LABOR CALCULATION section
            labor_section = self._extract_section(ai_text, "LABOR CALCULATION", "SUMMARY")
            if not labor_section:
                logger.warning("No LABOR CALCULATION section found in AI response")
                return LaborCalculation(hourly_rate=75.0, base_hours=8.0)  # Default values
            
            # Extract values using regex patterns
            base_hours_match = re.search(r"BASE HOURS:\s*(\d+\.\d+|\d+)", labor_section)
            hourly_rate_match = re.search(r"HOURLY RATE:\s*\$?(\d+\.\d+|\d+)", labor_section)
            complexity_factor_match = re.search(r"COMPLEXITY FACTOR:\s*(\d+\.\d+|\d+)", labor_section)
            notes_match = re.search(r"NOTES:\s*(.+?)(?=$)", labor_section, re.DOTALL)
            
            # Extract values from matches with defaults
            base_hours = float(base_hours_match.group(1)) if base_hours_match else 8.0
            hourly_rate = float(hourly_rate_match.group(1)) if hourly_rate_match else 75.0
            complexity_factor = float(complexity_factor_match.group(1)) if complexity_factor_match else 1.0
            notes = notes_match.group(1).strip() if notes_match else None
            
            return LaborCalculation(
                base_hours=base_hours,
                hourly_rate=hourly_rate,
                complexity_factor=complexity_factor,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error parsing labor calculation: {e}", exc_info=True)
            # Return default labor calculation rather than failing
            return LaborCalculation(hourly_rate=75.0, base_hours=8.0)

    def _extract_ai_confidence(self, ai_text: str) -> float:
        """
        Extract the AI's confidence score from the response.
        """
        try:
            # Look for CONFIDENCE in the SUMMARY section
            summary_section = self._extract_section(ai_text, "SUMMARY", "RECOMMENDATIONS")
            if not summary_section:
                return 0.85  # Default confidence if section not found
            
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)%", summary_section)
            if confidence_match:
                return float(confidence_match.group(1)) / 100  # Convert percentage to decimal
            return 0.85  # Default confidence if not found
            
        except Exception as e:
            logger.error(f"Error extracting AI confidence: {e}")
            return 0.85  # Default confidence on error

    def _generate_analysis_summary(self, ai_text: str) -> str:
        """
        Generate a concise summary of the AI analysis.
        """
        try:
            # Extract the SUMMARY section
            summary_section = self._extract_section(ai_text, "SUMMARY", "RECOMMENDATIONS")
            if summary_section:
                # Create a condensed summary
                lines = re.findall(r"^.*$", summary_section, re.MULTILINE)
                filtered_lines = [line.strip() for line in lines if line.strip() and ":" in line]
                if filtered_lines:
                    return " | ".join(filtered_lines)
            
            # If no summary section or empty, extract from recommendations
            recommendations = self._extract_section(ai_text, "RECOMMENDATIONS", None)
            if recommendations:
                # Take first few sentences
                sentences = re.split(r'[.!?]', recommendations)
                summary = '. '.join(sentence.strip() for sentence in sentences[:3] if sentence.strip()) + '.'
                return summary[:200] + ('...' if len(summary) > 200 else '')
            
            return "AI analysis completed. Estimate ready for review."
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return "AI analysis completed. Estimate ready for review."

    def _extract_section(self, text: str, section_name: str, next_section_name: Optional[str]) -> str:
        """
        Extract a section from the AI response text.
        
        Args:
            text: The full AI response text
            section_name: The name of the section to extract
            next_section_name: The name of the next section (or None if last section)
            
        Returns:
            The extracted section text or empty string if not found
        """
        try:
            # Create pattern to match section header, handling various formats
            section_pattern = rf"##?\s*{re.escape(section_name)}:?[\r\n]+"
            section_match = re.search(section_pattern, text, re.IGNORECASE)
            
            if not section_match:
                return ""
            
            start_idx = section_match.end()
            
            # Find the next section if specified
            if next_section_name:
                next_section_pattern = rf"##?\s*{re.escape(next_section_name)}:?[\r\n]+"
                next_section_match = re.search(next_section_pattern, text[start_idx:], re.IGNORECASE)
                
                if next_section_match:
                    end_idx = start_idx + next_section_match.start()
                    return text[start_idx:end_idx].strip()
            
            # If no next section or next section not found, take rest of text
            return text[start_idx:].strip()
            
        except Exception as e:
            logger.error(f"Error extracting section {section_name}: {e}")
            return ""

    def adjust_estimate_based_on_user_feedback(self, estimate: Estimate, user_feedback: Dict[str, Any]) -> Estimate:
        """
        Adjusts an estimate based on user feedback to improve future estimates.
        
        Args:
            estimate: The original estimate
            user_feedback: Dictionary containing user adjustments
            
        Returns:
            Updated estimate with user feedback incorporated
        """
        logger.info(f"Adjusting estimate based on user feedback")
        
        # Create a new version of the estimate
        updated_estimate = estimate.create_new_version()
        
        # Apply user adjustments to line items if provided
        if 'line_items' in user_feedback:
            for idx, item_feedback in user_feedback['line_items'].items():
                try:
                    idx = int(idx)
                    if idx < len(updated_estimate.line_items):
                        for key, value in item_feedback.items():
                            if hasattr(updated_estimate.line_items[idx], key):
                                setattr(updated_estimate.line_items[idx], key, value)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error applying line item feedback: {e}")
        
        # Apply user adjustments to labor calculation if provided
        if 'labor' in user_feedback:
            for key, value in user_feedback['labor'].items():
                if hasattr(updated_estimate.labor, key):
                    setattr(updated_estimate.labor, key, value)
        
        # Apply user adjustments to materials if provided
        if 'materials' in user_feedback:
            for idx, material_feedback in user_feedback['materials'].items():
                try:
                    idx = int(idx)
                    if idx < len(updated_estimate.materials):
                        for key, value in material_feedback.items():
                            if hasattr(updated_estimate.materials[idx], key):
                                setattr(updated_estimate.materials[idx], key, value)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error applying material feedback: {e}")
        
        # Update AI data to record user feedback
        if updated_estimate.ai_data:
            updated_estimate.ai_data.analysis_summary += " [User Adjusted]"
        
        return updated_estimate
```

## utils/dependency_injection.py
```python
"""
Lightweight dependency injection container for the application.
"""

from typing import Dict, Type, Any, Callable, Optional, get_type_hints
import inspect
from utils.logger import logger

class ServiceContainer:
    """
    Lightweight dependency injection container for managing application services.
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[Type, Any] = {}  # Key is service type, value is instance
        self._factories: Dict[Type, Callable] = {}  # Key is service type, value is factory function
        self._singletons: Dict[Type, bool] = {}  # Key is service type, value is singleton flag
    
    def register(self, service_type: Type, instance: Any = None, factory: Callable = None, singleton: bool = True):
        """
        Register a service with the container.
        
        Args:
            service_type: The type (class) of the service
            instance: An optional pre-created instance of the service
            factory: An optional factory function to create the service
            singleton: Whether the service should be a singleton (True) or transient (False)
        """
        if instance is not None and factory is not None:
            raise ValueError("Cannot specify both instance and factory")
        
        if instance is not None:
            self._services[service_type] = instance
        else:
            self._factories[service_type] = factory or service_type
            self._singletons[service_type] = singleton
            
            # If it's a singleton and we have a factory, pre-create the instance
            if singleton and service_type not in self._services:
                self._services[service_type] = None  # Mark for lazy initialization
    
    def get(self, service_type: Type) -> Any:
        """
        Get an instance of the requested service type.
        
        Args:
            service_type: The type (class) of the service to retrieve
            
        Returns:
            An instance of the requested service
            
        Raises:
            KeyError: If the service type is not registered
        """
        # If service is already instantiated, return it
        if service_type in self._services and self._services[service_type] is not None:
            return self._services[service_type]
        
        # If we have a factory for this service
        if service_type in self._factories:
            # Create the service
            instance = self._create_instance(service_type)
            
            # If it's a singleton, store it
            if self._singletons.get(service_type, False):
                self._services[service_type] = instance
                
            return instance
        
        raise KeyError(f"Service type {service_type.__name__} not registered")
    
    def _create_instance(self, service_type: Type) -> Any:
        """
        Create an instance of the service, resolving dependencies.
        
        Args:
            service_type: The type of service to create
            
        Returns:
            An instance of the service with dependencies injected
        """
        factory = self._factories[service_type]
        
        # If factory is a class (the service type itself)
        if inspect.isclass(factory):
            # Get constructor parameter types
            try:
                constructor = factory.__init__
                type_hints = get_type_hints(constructor)
                
                # Remove return type hint if present
                if 'return' in type_hints:
                    del type_hints['return']
                
                # Get parameter names
                signature = inspect.signature(constructor)
                params = {}
                
                # For each parameter (excluding self)
                for name, param in signature.parameters.items():
                    if name == 'self':
                        continue
                    
                    # If parameter has a type hint and we can resolve it
                    if name in type_hints:
                        param_type = type_hints[name]
                        try:
                            params[name] = self.get(param_type)
                        except KeyError:
                            # If parameter has a default value, use it
                            if param.default is not inspect.Parameter.empty:
                                params[name] = param.default
                            else:
                                logger.warning(f"Cannot resolve dependency {name}: {param_type} for {service_type.__name__}")
                
                return factory(**params)
            except Exception as e:
                logger.error(f"Error creating instance of {service_type.__name__}: {e}")
                return factory()  # Fallback to creating without dependencies
        else:
            # Factory is a custom factory function
            return factory(self)
    
    def create_scope(self):
        """
        Create a new container scope that inherits singleton services but creates new transient services.
        
        Returns:
            A new ServiceContainer with inherited singleton services
        """
        scope = ServiceContainer()
        
        # Copy singleton services to the new scope
        for service_type, instance in self._services.items():
            if self._singletons.get(service_type, False):
                scope._services[service_type] = instance
        
        # Copy factories for transient services
        for service_type, factory in self._factories.items():
            if not self._singletons.get(service_type, False):
                scope._factories[service_type] = factory
                scope._singletons[service_type] = False
        
        return scope

class ServiceLocator:
    """
    Service locator for components that need dynamic access to services.
    """
    
    _instance = None
    _container = None
    
    @classmethod
    def initialize(cls, container: ServiceContainer):
        """
        Initialize the service locator with a container.
        
        Args:
            container: The service container to use
        """
        cls._instance = cls()
        cls._container = container
    
    @classmethod
    def get_instance(cls):
        """
        Get the service locator instance.
        
        Returns:
            The service locator instance
            
        Raises:
            RuntimeError: If the service locator has not been initialized
        """
        if cls._instance is None:
            raise RuntimeError("ServiceLocator has not been initialized")
        return cls._instance
    
    @classmethod
    def get(cls, service_type: Type) -> Any:
        """
        Get a service from the container.
        
        Args:
            service_type: The type of service to retrieve
            
        Returns:
            An instance of the requested service
            
        Raises:
            RuntimeError: If the service locator has not been initialized
        """
        if cls._container is None:
            raise RuntimeError("ServiceLocator has not been initialized")
        return cls._container.get(service_type)
```

## utils/error_handling.py
```python
"""
Centralized error handling system for the application.
"""

import sys
import traceback
import json
from typing import Dict, Any, Optional, Callable, Type, List, Union
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget
from utils.logger import logger
from utils.notification_manager import NotificationManager

class ContextualError(Exception):
    """Base exception class with contextual information for better debugging."""
    
    def __init__(self, message: str, context: Dict[str, Any] = None, cause: Exception = None):
        """
        Initialize a contextual error.
        
        Args:
            message: Error message
            context: Dictionary of contextual information
            cause: The original exception that caused this error
        """
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Construct error message with context information
        full_message = message
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} [Context: {context_str}]"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging or serialization."""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
        
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return result

# Domain-specific exception classes
class AIServiceError(ContextualError):
    """Error occurred in the AI service."""
    pass

class ComputerVisionError(ContextualError):
    """Error occurred in the computer vision service."""
    pass

class MeasurementError(ContextualError):
    """Error occurred in the measurement service."""
    pass

class DatabaseError(ContextualError):
    """Error occurred in the database layer."""
    pass

class AIParsingError(ContextualError):
    """Error occurred during AI response parsing."""
    pass

class EstimateGenerationError(ContextualError):
    """Error occurred during estimate generation."""
    pass

class ErrorHandler:
    """
    Centralized error handler for the application.
    """
    
    _instance = None
    _error_strategies: Dict[Type[Exception], Callable] = {}
    _notification_manager: Optional[NotificationManager] = None
    
    @classmethod
    def initialize(cls, notification_manager: NotificationManager = None):
        """
        Initialize the error handler.
        
        Args:
            notification_manager: Optional notification manager for displaying errors
        """
        if cls._instance is None:
            cls._instance = cls()
        cls._notification_manager = notification_manager
        
        # Set up global exception hook
        sys.excepthook = cls.global_exception_handler
    
    @classmethod
    def register_strategy(cls, exception_type: Type[Exception], strategy: Callable):
        """
        Register an error handling strategy for a specific exception type.
        
        Args:
            exception_type: The type of exception to handle
            strategy: A callable that will handle the exception
        """
        cls._error_strategies[exception_type] = strategy
    
    @classmethod
    def handle_error(cls, exception: Exception, context: Dict[str, Any] = None) -> bool:
        """
        Handle an exception with the appropriate strategy.
        
        Args:
            exception: The exception to handle
            context: Optional context information
            
        Returns:
            True if the error was handled, False otherwise
        """
        # Log the error
        cls._log_error(exception, context)
        
        # Try specific strategies first
        for exception_type, strategy in cls._error_strategies.items():
            if isinstance(exception, exception_type):
                return strategy(exception, context)
        
        # Default handling
        return cls._default_error_handler(exception, context)
    
    @classmethod
    def _log_error(cls, exception: Exception, context: Dict[str, Any] = None):
        """
        Log an error with context information.
        
        Args:
            exception: The exception to log
            context: Optional context information
        """
        if isinstance(exception, ContextualError):
            error_dict = exception.to_dict()
            if context:
                error_dict["additional_context"] = context
            logger.error(f"Error occurred: {json.dumps(error_dict)}")
        else:
            # For standard exceptions, log the error with traceback
            error_type = type(exception).__name__
            error_message = str(exception)
            
            context_str = ""
            if context:
                context_str = f" [Context: {json.dumps(context)}]"
            
            logger.error(f"{error_type}: {error_message}{context_str}", exc_info=True)
    
    @classmethod
    def _default_error_handler(cls, exception: Exception, context: Dict[str, Any] = None) -> bool:
        """
        Default error handling strategy.
        
        Args:
            exception: The exception to handle
            context: Optional context information
            
        Returns:
            True if the error was handled, False otherwise
        """
        # If we have a notification manager, show notification
        if cls._notification_manager:
            error_message = str(exception)
            error_type = type(exception).__name__
            
            cls._notification_manager.notify(
                title=f"Error: {error_type}",
                message=error_message,
                level="error",
                show_popup=True
            )
            return True
        
        return False
    
    @classmethod
    def global_exception_handler(cls, exc_type, exc_value, exc_traceback):
        """
        Global exception handler for uncaught exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Don't handle KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the error
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Try to show an error dialog
        try:
            app = QApplication.instance()
            if app is not None:
                error_msg = QMessageBox()
                error_msg.setIcon(QMessageBox.Critical)
                error_msg.setText("An unhandled error occurred")
                error_msg.setInformativeText(str(exc_value))
                error_msg.setDetailedText("\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
                error_msg.setWindowTitle("Application Error")
                error_msg.exec_()
        except Exception:
            pass
        
        # Optional: terminate the application
        # sys.exit(1)

class ErrorBoundary(QWidget):
    """
    Error boundary component for UI error isolation.
    """
    
    def __init__(self, parent=None, fallback_widget: Optional[QWidget] = None):
        """
        Initialize the error boundary.
        
        Args:
            parent: Parent widget
            fallback_widget: Optional widget to display in case of error
        """
        super().__init__(parent)
        self._child_widgets: List[QWidget] = []
        self._fallback_widget = fallback_widget
        self._has_error = False
    
    def add_widget(self, widget: QWidget):
        """
        Add a widget to be protected by the error boundary.
        
        Args:
            widget: The widget to protect
        """
        self._child_widgets.append(widget)
        widget.setParent(self)
    
    def handle_error(self, error: Exception):
        """
        Handle an error in a child widget.
        
        Args:
            error: The error that occurred
        """
        self._has_error = True
        
        # Log the error
        logger.error(f"Error in UI component: {error}", exc_info=True)
        
        # Hide all child widgets
        for widget in self._child_widgets:
            widget.hide()
        
        # Show fallback widget if available
        if self._fallback_widget:
            self._fallback_widget.setParent(self)
            self._fallback_widget.show()
        else:
            # Create a basic error message
            from PyQt5.QtWidgets import QLabel, QVBoxLayout
            
            error_layout = QVBoxLayout(self)
            error_label = QLabel(f"An error occurred in this component:\n{error}")
            error_label.setStyleSheet("color: red; background-color: #ffeeee; padding: 10px;")
            error_layout.addWidget(error_label)
            
            retry_button = QPushButton("Retry")
            retry_button.clicked.connect(self.reset)
            error_layout.addWidget(retry_button)
            
            self.setLayout(error_layout)
    
    def reset(self):
        """Reset the error boundary and show child widgets again."""
        if not self._has_error:
            return
        
        # Remove fallback widget
        if self._fallback_widget:
            self._fallback_widget.hide()
        
        # Clear any error layout
        while self.layout():
            old_layout = self.layout()
            for i in reversed(range(old_layout.count())):
                widget = old_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            old_layout.setParent(None)
        
        # Show all child widgets
        for widget in self._child_widgets:
            widget.show()
        
        self._has_error = False
```

## ui/vision/measurement_canvas.py
```python
"""
Interactive canvas for displaying and manipulating measurements on images.
"""

from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                            QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsTextItem,
                            QMenu, QAction, QInputDialog, QGraphicsPathItem)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QPainterPath, QFont, QCursor
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF, pyqtSignal, QSizeF
from typing import List, Dict, Tuple, Union, Optional, Any, Set
import math
import numpy as np
from enum import Enum

class MeasurementMode(Enum):
    """Measurement interaction modes."""
    VIEWING = 0
    CALIBRATION = 1
    LENGTH_MEASUREMENT = 2
    PATH_MEASUREMENT = 3
    AREA_MEASUREMENT = 4
    ANNOTATION = 5

class MeasurementCanvas(QGraphicsView):
    """
    Interactive canvas for displaying images and measurements.
    """
    # Signals for measurement updates
    calibration_updated = pyqtSignal(float, str)  # pixels_per_unit, unit
    measurement_created = pyqtSignal(Dict[str, Any])  # measurement data
    measurement_updated = pyqtSignal(Dict[str, Any])  # measurement data
    measurement_deleted = pyqtSignal(int)  # measurement id

    def __init__(self, parent=None):
        """Initializes the MeasurementCanvas."""
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)  # Set image to background
        self.scene.addItem(self.image_item)

        # Canvas state
        self.scale_factor = 1.0  # Initial scale factor for zoom
        self.measurement_mode = MeasurementMode.VIEWING
        self.current_image_path = None
        self.pixels_per_unit = None
        self.measurement_unit = "inches"  # Default unit
        self.calibration_reference_length = 0  # Reference length in real-world units

        # Current measurement in progress
        self.current_points = []
        self.temp_items = []  # Temporary items to visualize current measurement
        self.selected_measurement_id = None
        
        # Collections of measurements
        self.next_measurement_id = 1
        self.measurements = {}  # id -> measurement data
        self.measurement_items = {}  # id -> graphical items
        
        # Edge detection and snapping
        self.detected_edges = []  # [(x1, y1, x2, y2), ...]
        self.enable_edge_snapping = True
        self.snapping_threshold = 15  # pixels
        
        # Drawing styles
        self.measurement_styles = {
            "length": {
                "pen": QPen(QColor(0, 120, 255), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin),
                "endpoint_pen": QPen(QColor(0, 80, 200), 2, Qt.SolidLine),
                "endpoint_brush": QBrush(QColor(255, 255, 255)),
                "text_color": QColor(0, 80, 200),
                "handle_size": 10  # Size of endpoints for user interaction
            },
            "path": {
                "pen": QPen(QColor(0, 200, 0), 2.5, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin),
                "point_pen": QPen(QColor(0, 150, 0), 2, Qt.SolidLine),
                "point_brush": QBrush(QColor(255, 255, 255)),
                "text_color": QColor(0, 150, 0),
                "handle_size": 8
            },
            "area": {
                "pen": QPen(QColor(255, 100, 0), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin),
                "fill_brush": QBrush(QColor(255, 165, 0, 50)),  # Semi-transparent
                "text_color": QColor(200, 80, 0),
                "handle_size": 8
            },
            "annotation": {
                "pen": QPen(QColor(200, 0, 100), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin),
                "text_color": QColor(200, 0, 100),
                "handle_size": 8
            },
            "calibration": {
                "pen": QPen(QColor(0, 170, 0), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin),
                "endpoint_pen": QPen(QColor(0, 120, 0), 2, Qt.SolidLine),
                "endpoint_brush": QBrush(QColor(255, 255, 255)),
                "text_color": QColor(0, 120, 0),
                "handle_size": 10
            }
        }
        
        # Additional visual settings
        self.highlight_selected = True
        self.show_measurement_labels = True
        self.show_grid = False
        self.grid_spacing = 50  # pixels

        # Zoom and pan settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning by default

    def load_image(self, image_path: str):
        """
        Loads an image onto the canvas.
        
        Args:
            image_path: Path to the image file.
        """
        self.current_image_path = image_path
        try:
            image = QImage(image_path)
            if image.isNull():
                raise ValueError(f"Could not load image: {image_path}")
                
            pixmap = QPixmap.fromImage(image)
            self.image_item.setPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))  # Adjust scene size to image
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)  # Initial fit to view
            
            # Reset current state
            self.clear_temp_items()
            self.current_points = []
            self.scale_factor = 1.0
            
            # Run edge detection if enabled
            self.detect_edges()
            
            return True
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False

    def set_measurement_mode(self, mode: MeasurementMode):
        """
        Sets the measurement interaction mode.
        
        Args:
            mode: The measurement mode to set
        """
        self.measurement_mode = mode
        self.clear_temp_items()
        self.current_points = []
        
        # Set appropriate cursor based on mode
        if mode == MeasurementMode.VIEWING:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            if mode == MeasurementMode.CALIBRATION:
                self.viewport().setCursor(Qt.CrossCursor)
            elif mode == MeasurementMode.LENGTH_MEASUREMENT:
                self.viewport().setCursor(QCursor(QPixmap("icons/ruler_cursor.png")))
            elif mode == MeasurementMode.PATH_MEASUREMENT:
                self.viewport().setCursor(QCursor(QPixmap("icons/path_cursor.png")))
            elif mode == MeasurementMode.AREA_MEASUREMENT:
                self.viewport().setCursor(QCursor(QPixmap("icons/area_cursor.png")))
            elif mode == MeasurementMode.ANNOTATION:
                self.viewport().setCursor(QCursor(QPixmap("icons/annotation_cursor.png")))
            else:
                self.viewport().setCursor(Qt.CrossCursor)

    def set_calibration(self, pixels_per_unit: float, unit: str = "inches"):
        """
        Sets the measurement calibration directly.
        
        Args:
            pixels_per_unit: Number of pixels per unit
            unit: Unit of measurement (inches, cm, etc.)
        """
        self.pixels_per_unit = pixels_per_unit
        self.measurement_unit = unit
        self.calibration_updated.emit(pixels_per_unit, unit)
        self.update_all_measurement_labels()

    def detect_edges(self):
        """
        Detects edges in the current image for smart snapping.
        This is a placeholder that should be replaced with actual edge detection.
        In a production app, this would use OpenCV or similar.
        """
        # Placeholder for edge detection - this would use CV techniques in production
        # For now, just detect the image borders as edges
        if self.image_item.pixmap():
            rect = self.image_item.pixmap().rect()
            
            # Add image borders as edges
            self.detected_edges = [
                # Top edge
                (rect.left(), rect.top(), rect.right(), rect.top()),
                # Right edge
                (rect.right(), rect.top(), rect.right(), rect.bottom()),
                # Bottom edge
                (rect.right(), rect.bottom(), rect.left(), rect.bottom()),
                # Left edge
                (rect.left(), rect.bottom(), rect.left(), rect.top()),
            ]
            
            # In a real implementation, you would run edge detection on the image
            # and add detected lines to self.detected_edges

    def mousePressEvent(self, event):
        """
        Handles mouse press events for interactive features.
        
        Args:
            event: Mouse event
        """
        if not self.image_item.pixmap():
            super().mousePressEvent(event)
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        # Handle right click for context menu
        if event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())
            return
            
        # If in viewing mode, check if we clicked on a measurement handle
        if self.measurement_mode == MeasurementMode.VIEWING:
            item = self.itemAt(event.pos())
            if item and hasattr(item, 'measurement_id'):
                self.selected_measurement_id = item.measurement_id
                self.update_measurement_appearance()
                super().mousePressEvent(event)
                return
        
        # Handle different measurement modes
        snapped_pos = self.snap_to_edge(scene_pos) if self.enable_edge_snapping else scene_pos
        
        if self.measurement_mode == MeasurementMode.CALIBRATION:
            self.handle_calibration_click(snapped_pos)
        elif self.measurement_mode == MeasurementMode.LENGTH_MEASUREMENT:
            self.handle_length_measurement_click(snapped_pos)
        elif self.measurement_mode == MeasurementMode.PATH_MEASUREMENT:
            self.handle_path_measurement_click(snapped_pos)
        elif self.measurement_mode == MeasurementMode.AREA_MEASUREMENT:
            self.handle_area_measurement_click(snapped_pos)
        elif self.measurement_mode == MeasurementMode.ANNOTATION:
            self.handle_annotation_click(snapped_pos)
        
        super().mousePressEvent(event)

    def handle_calibration_click(self, pos: QPointF):
        """
        Handles clicks in calibration mode.
        
        Args:
            pos: Click position in scene coordinates
        """
        if len(self.current_points) < 2:
            self.current_points.append(pos)
            self.update_calibration_visualization()
            
            # If we have two points, prompt for the reference length
            if len(self.current_points) == 2:
                p1, p2 = self.current_points
                pixel_distance = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                
                # Prompt user for reference length
                reference_length, ok = QInputDialog.getDouble(
                    self, 
                    "Calibration",
                    f"Enter the real-world length in {self.measurement_unit}:",
                    value=10.0,
                    min=0.1,
                    max=1000.0,
                    decimals=2
                )
                
                if ok and reference_length > 0:
                    # Calculate pixels per unit
                    self.pixels_per_unit = pixel_distance / reference_length
                    self.calibration_reference_length = reference_length
                    
                    # Emit signal and reset state
                    self.calibration_updated.emit(self.pixels_per_unit, self.measurement_unit)
                    self.update_all_measurement_labels()
                    
                    # Create a persistent calibration visualization
                    self.create_calibration_marker()
                
                # Reset state
                self.clear_temp_items()
                self.current_points = []
                self.set_measurement_mode(MeasurementMode.VIEWING)

    def handle_length_measurement_click(self, pos: QPointF):
        """
        Handles clicks in length measurement mode.
        
        Args:
            pos: Click position in scene coordinates
        """
        if len(self.current_points) < 2:
            self.current_points.append(pos)
            self.update_length_measurement_visualization()
            
            # If we have two points, create the measurement
            if len(self.current_points) == 2:
                self.create_length_measurement()
                
                # Reset state for another measurement
                self.clear_temp_items()
                self.current_points = []

    def handle_path_measurement_click(self, pos: QPointF):
        """
        Handles clicks in path measurement mode.
        
        Args:
            pos: Click position in scene coordinates
        """
        # Add point to the path
        self.current_points.append(pos)
        self.update_path_measurement_visualization()
        
        # Double-click ends the path
        if len(self.current_points) > 1 and self.is_double_click():
            self.create_path_measurement()
            
            # Reset state for another measurement
            self.clear_temp_items()
            self.current_points = []

    def handle_area_measurement_click(self, pos: QPointF):
        """
        Handles clicks in area measurement mode.
        
        Args:
            pos: Click position in scene coordinates
        """
        # Add point to the area boundary
        self.current_points.append(pos)
        self.update_area_measurement_visualization()
        
        # Double-click ends the area
        if len(self.current_points) > 2 and self.is_double_click():
            self.create_area_measurement()
            
            # Reset state for another measurement
            self.clear_temp_items()
            self.current_points = []

    def handle_annotation_click(self, pos: QPointF):
        """
        Handles clicks in annotation mode.
        
        Args:
            pos: Click position in scene coordinates
        """
        # In annotation mode, a single click creates an annotation point
        text, ok = QInputDialog.getText(
            self, 
            "Annotation",
            "Enter annotation text:",
            text=""
        )
        
        if ok and text:
            self.create_annotation(pos, text)

    def is_double_click(self) -> bool:
        """
        Checks if we are processing a double-click.
        This is a simple placeholder; a real implementation would check time between clicks.
        
        Returns:
            True if this is a double-click, False otherwise
        """
        # Placeholder for double-click detection
        # In practice, track click times or use QApplication.doubleClickInterval()
        return False

    def snap_to_edge(self, pos: QPointF) -> QPointF:
        """
        Snaps a point to nearby detected edges if within threshold.
        
        Args:
            pos: Original position
            
        Returns:
            Snapped position or original if no snap
        """
        if not self.enable_edge_snapping or not self.detected_edges:
            return pos
        
        closest_dist = float('inf')
        closest_point = pos
        
        for x1, y1, x2, y2 in self.detected_edges:
            # Convert to QLineF for easier calculation
            line = QLineF(x1, y1, x2, y2)
            
            # Project the point onto the line
            line_dir = QPointF(line.dx(), line.dy())
            line_length_squared = line.dx()**2 + line.dy()**2
            
            if line_length_squared < 1e-10:  # Avoid division by zero
                continue
                
            t = max(0, min(1, QPointF.dotProduct(
                QPointF(pos.x() - line.x1(), pos.y() - line.y1()),
                line_dir
            ) / line_length_squared))
            
            projection = QPointF(
                line.x1() + t * line.dx(),
                line.y1() + t * line.dy()
            )
            
            # Calculate distance to the projection
            dist = math.sqrt((pos.x() - projection.x())**2 + (pos.y() - projection.y())**2)
            
            if dist < self.snapping_threshold and dist < closest_dist:
                closest_dist = dist
                closest_point = projection
        
        return closest_point

    def update_calibration_visualization(self):
        """Updates the canvas to visualize calibration points and lines."""
        self.clear_temp_items()
        
        style = self.measurement_styles["calibration"]
        
        if len(self.current_points) > 0:
            # Draw first point
            p1 = self.current_points[0]
            handle_size = style["handle_size"]
            
            endpoint1 = QGraphicsEllipseItem(
                p1.x() - handle_size/2, p1.y() - handle_size/2,
                handle_size, handle_size
            )
            endpoint1.setPen(style["endpoint_pen"])
            endpoint1.setBrush(style["endpoint_brush"])
            self.scene.addItem(endpoint1)
            self.temp_items.append(endpoint1)
            
            if len(self.current_points) == 2:
                # Draw second point and connecting line
                p2 = self.current_points[1]
                
                line = QGraphicsLineItem(QLineF(p1, p2))
                line.setPen(style["pen"])
                self.scene.addItem(line)
                self.temp_items.append(line)
                
                endpoint2 = QGraphicsEllipseItem(
                    p2.x() - handle_size/2, p2.y() - handle_size/2,
                    handle_size, handle_size
                )
                endpoint2.setPen(style["endpoint_pen"])
                endpoint2.setBrush(style["endpoint_brush"])
                self.scene.addItem(endpoint2)
                self.temp_items.append(endpoint2)
                
                # Display the distance
                pixel_distance = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                mid_point = QPointF((p1.x() + p2.x())/2, (p1.y() + p2.y())/2)
                
                text_item = QGraphicsTextItem(f"{pixel_distance:.1f} pixels")
                text_item.setDefaultTextColor(style["text_color"])
                text_item.setFont(QFont("Arial", 10, QFont.Bold))
                
                # Position text above the line
                text_rect = text_item.boundingRect()
                text_item.setPos(
                    mid_point.x() - text_rect.width()/2,
                    mid_point.y() - text_rect.height() - 5
                )
                
                self.scene.addItem(text_item)
                self.temp_items.append(text_item)

    def update_length_measurement_visualization(self):
        """Updates the canvas to visualize length measurement in progress."""
        self.clear_temp_items()
        
        style = self.measurement_styles["length"]
        
        if len(self.current_points) > 0:
            # Draw first point
            p1 = self.current_points[0]
            handle_size = style["handle_size"]
            
            endpoint1 = QGraphicsEllipseItem(
                p1.x() - handle_size/2, p1.y() - handle_size/2,
                handle_size, handle_size
            )
            endpoint1.setPen(style["endpoint_pen"])
            endpoint1.setBrush(style["endpoint_brush"])
            self.scene.addItem(endpoint1)
            self.temp_items.append(endpoint1)
            
            if len(self.current_points) == 2:
                # Draw second point and connecting line
                p2 = self.current_points[1]
                
                line = QGraphicsLineItem(QLineF(p1, p2))
                line.setPen(style["pen"])
                self.scene.addItem(line)
                self.temp_items.append(line)
                
                endpoint2 = QGraphicsEllipseItem(
                    p2.x() - handle_size/2, p2.y() - handle_size/2,
                    handle_size, handle_size
                )
                endpoint2.setPen(style["endpoint_pen"])
                endpoint2.setBrush(style["endpoint_brush"])
                self.scene.addItem(endpoint2)
                self.temp_items.append(endpoint2)
                
                # Display the distance
                if self.pixels_per_unit:
                    pixel_distance = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                    real_distance = pixel_distance / self.pixels_per_unit
                    mid_point = QPointF((p1.x() + p2.x())/2, (p1.y() + p2.y())/2)
                    
                    text_item = QGraphicsTextItem(f"{real_distance:.2f} {self.measurement_unit}")
                    text_item.setDefaultTextColor(style["text_color"])
                    text_item.setFont(QFont("Arial", 10, QFont.Bold))
                    
                    # Position text above the line
                    text_rect = text_item.boundingRect()
                    text_item.setPos(
                        mid_point.x() - text_rect.width()/2,
                        mid_point.y() - text_rect.height() - 5
                    )
                    
                    self.scene.addItem(text_item)
                    self.temp_items.append(text_item)

    def update_path_measurement_visualization(self):
        """Updates the canvas to visualize path measurement in progress."""
        self.clear_temp_items()
        
        style = self.measurement_styles["path"]
        
        if len(self.current_points) > 0:
            # Create a path connecting all points
            path = QPainterPath()
            path.moveTo(self.current_points[0])
            
            for i in range(1, len(self.current_points)):
                path.lineTo(self.current_points[i])
            
            # Draw the path
            path_item = QGraphicsPathItem(path)
            path_item.setPen(style["pen"])
            self.scene.addItem(path_item)
            self.temp_items.append(path_item)
            
            # Draw points along the path
            handle_size = style["handle_size"]
            
            for point in self.current_points:
                point_item = QGraphicsEllipseItem(
                    point.x() - handle_size/2, point.y() - handle_size/2,
                    handle_size, handle_size
                )
                point_item.setPen(style["point_pen"])
                point_item.setBrush(style["point_brush"])
                self.scene.addItem(point_item)
                self.temp_items.append(point_item)
            
            # Display the total path length
            if self.pixels_per_unit and len(self.current_points) > 1:
                total_pixel_length = 0
                
                for i in range(1, len(self.current_points)):
                    p1 = self.current_points[i-1]
                    p2 = self.current_points[i]
                    segment_length = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                    total_pixel_length += segment_length
                
                real_length = total_pixel_length / self.pixels_per_unit
                
                # Place the text near the last point
                last_point = self.current_points[-1]
                
                text_item = QGraphicsTextItem(f"Path: {real_length:.2f} {self.measurement_unit}")
                text_item.setDefaultTextColor(style["text_color"])
                text_item.setFont(QFont("Arial", 10, QFont.Bold))
                
                # Position text near the last point
                text_rect = text_item.boundingRect()
                text_item.setPos(
                    last_point.x() + 10,
                    last_point.y() - text_rect.height()/2
                )
                
                self.scene.addItem(text_item)
                self.temp_items.append(text_item)

    def update_area_measurement_visualization(self):
        """Updates the canvas to visualize area measurement in progress."""
        self.clear_temp_items()
        
        style = self.measurement_styles["area"]
        
        if len(self.current_points) > 0:
            # Create a polygon path
            path = QPainterPath()
            path.moveTo(self.current_points[0])
            
            for i in range(1, len(self.current_points)):
                path.lineTo(self.current_points[i])
            
            # Close the path if we have more than 2 points
            if len(self.current_points) > 2:
                path.closeSubpath()
            
            # Draw the polygon
            path_item = QGraphicsPathItem(path)
            path_item.setPen(style["pen"])
            path_item.setBrush(style["fill_brush"])
            self.scene.addItem(path_item)
            self.temp_items.append(path_item)
            
            # Draw points
            handle_size = style["handle_size"]
            
            for point in self.current_points:
                point_item = QGraphicsEllipseItem(
                    point.x() - handle_size/2, point.y() - handle_size/2,
                    handle_size, handle_size
                )
                point_item.setPen(QPen(style["pen"].color()))
                point_item.setBrush(QBrush(Qt.white))
                self.scene.addItem(point_item)
                self.temp_items.append(point_item)
            
            # Display the area if we have a closed polygon
            if self.pixels_per_unit and len(self.current_points) > 2:
                # Calculate polygon area in pixels
                pixel_area = self.calculate_polygon_area(self.current_points)
                
                # Convert to real-world area
                real_area = pixel_area / (self.pixels_per_unit ** 2)
                
                # Calculate centroid for text placement
                centroid_x = sum(p.x() for p in self.current_points) / len(self.current_points)
                centroid_y = sum(p.y() for p in self.current_points) / len(self.current_points)
                
                text_item = QGraphicsTextItem(f"Area: {real_area:.2f} {self.measurement_unit}²")
                text_item.setDefaultTextColor(style["text_color"])
                text_item.setFont(QFont("Arial", 10, QFont.Bold))
                
                # Position text at centroid
                text_rect = text_item.boundingRect()
                text_item.setPos(
                    centroid_x - text_rect.width()/2,
                    centroid_y - text_rect.height()/2
                )
                
                self.scene.addItem(text_item)
                self.temp_items.append(text_item)

    def create_calibration_marker(self):
        """Creates a persistent calibration marker on the canvas."""
        if len(self.current_points) != 2:
            return
        
        p1, p2 = self.current_points
        style = self.measurement_styles["calibration"]
        
        # Create a group of items for this measurement
        items = []
        
        # Create line
        line = QGraphicsLineItem(QLineF(p1, p2))
        line.setPen(style["pen"])
        line.setZValue(10)  # Above image
        self.scene.addItem(line)
        items.append(line)
        
        # Create endpoints
        handle_size = style["handle_size"]
        
        for point in [p1, p2]:
            endpoint = QGraphicsEllipseItem(
                point.x() - handle_size/2, point.y() - handle_size/2,
                handle_size, handle_size
            )
            endpoint.setPen(style["endpoint_pen"])
            endpoint.setBrush(style["endpoint_brush"])
            endpoint.setZValue(11)  # Above lines
            self.scene.addItem(endpoint)
            items.append(endpoint)
        
        # Create label
        mid_point = QPointF((p1.x() + p2.x())/2, (p1.y() + p2.y())/2)
        
        text_item = QGraphicsTextItem(
            f"Calibration: {self.calibration_reference_length} {self.measurement_unit} "
            f"({self.pixels_per_unit:.2f} px/{self.measurement_unit})"
        )
        text_item.setDefaultTextColor(style["text_color"])
        text_item.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Position text above the line
        text_rect = text_item.boundingRect()
        text_item.setPos(
            mid_point.x() - text_rect.width()/2,
            mid_point.y() - text_rect.height() - 5
        )
        text_item.setZValue(11)  # Above lines
        
        self.scene.addItem(text_item)
        items.append(text_item)
        
        # Tag items with measurement_id for selection and store in measurements
        measurement_id = self.next_measurement_id
        
        measurement_data = {
            'id': measurement_id,
            'type': 'calibration',
            'points': [(p.x(), p.y()) for p in self.current_points],
            'reference_length': self.calibration_reference_length,
            'unit': self.measurement_unit,
            'pixels_per_unit': self.pixels_per_unit
        }
        
        for item in items:
            item.measurement_id = measurement_id
        
        self.measurements[measurement_id] = measurement_data
        self.measurement_items[measurement_id] = items
        self.next_measurement_id += 1

    def create_length_measurement(self):
        """Creates a persistent length measurement on the canvas."""
        if len(self.current_points) != 2 or not self.pixels_per_unit:
            return
        
        p1, p2 = self.current_points
        style = self.measurement_styles["length"]
        
        # Create a group of items for this measurement
        items = []
        
        # Create line
        line = QGraphicsLineItem(QLineF(p1, p2))
        line.setPen(style["pen"])
        line.setZValue(10)  # Above image
        self.scene.addItem(line)
        items.append(line)
        
        # Create endpoints (handles)
        handle_size = style["handle_size"]
        
        for point in [p1, p2]:
            endpoint = QGraphicsEllipseItem(
                point.x() - handle_size/2, point.y() - handle_size/2,
                handle_size, handle_size
            )
            endpoint.setPen(style["endpoint_pen"])
            endpoint.setBrush(style["endpoint_brush"])
            endpoint.setZValue(11)  # Above lines
            self.scene.addItem(endpoint)
            items.append(endpoint)
        
        # Create label with distance
        pixel_distance = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
        real_distance = pixel_distance / self.pixels_per_unit
        
        mid_point = QPointF((p1.x() + p2.x())/2, (p1.y() + p2.y())/2)
        
        text_item = QGraphicsTextItem(f"{real_distance:.2f} {self.measurement_unit}")
        text_item.setDefaultTextColor(style["text_color"])
        text_item.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Position text above the line
        text_rect = text_item.boundingRect()
        text_item.setPos(
            mid_point.x() - text_rect.width()/2,
            mid_point.y() - text_rect.height() - 5
        )
        text_item.setZValue(11)  # Above lines
        
        self.scene.addItem(text_item)
        items.append(text_item)
        
        # Tag items with measurement_id for selection and store in measurements
        measurement_id = self.next_measurement_id
        
        measurement_data = {
            'id': measurement_id,
            'type': 'length',
            'points': [(p.x(), p.y()) for p in self.current_points],
            'pixel_length': pixel_distance,
            'real_length': real_distance,
            'unit': self.measurement_unit
        }
        
        for item in items:
            item.measurement_id = measurement_id
        
        self.measurements[measurement_id] = measurement_data
        self.measurement_items[measurement_id] = items
        self.next_measurement_id += 1
        
        # Emit signal with measurement data
        self.measurement_created.emit(measurement_data)

    def create_path_measurement(self):
        """Creates a persistent path measurement on the canvas."""
        if len(self.current_points) < 2 or not self.pixels_per_unit:
            return
        
        style = self.measurement_styles["path"]
        
        # Create a group of items for this measurement
        items = []
        
        # Create the path
        path = QPainterPath()
        path.moveTo(self.current_points[0])
        
        for i in range(1, len(self.current_points)):
            path.lineTo(self.current_points[i])
        
        path_item = QGraphicsPathItem(path)
        path_item.setPen(style["pen"])
        path_item.setZValue(10)  # Above image
        self.scene.addItem(path_item)
        items.append(path_item)
        
        # Create points along the path
        handle_size = style["handle_size"]
        
        for point in self.current_points:
            point_item = QGraphicsEllipseItem(
                point.x() - handle_size/2, point.y() - handle_size/2,
                handle_size, handle_size
            )
            point_item.setPen(style["point_pen"])
            point_item.setBrush(style["point_brush"])
            point_item.setZValue(11)  # Above lines
            self.scene.addItem(point_item)
            items.append(point_item)
        
        # Calculate the total path length
        total_pixel_length = 0
        
        for i in range(1, len(self.current_points)):
            p1 = self.current_points[i-1]
            p2 = self.current_points[i]
            segment_length = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
            total_pixel_length += segment_length
        
        real_length = total_pixel_length / self.pixels_per_unit
        
        # Create label with path length
        last_point = self.current_points[-1]
        
        text_item = QGraphicsTextItem(f"Path: {real_length:.2f} {self.measurement_unit}")
        text_item.setDefaultTextColor(style["text_color"])
        text_item.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Position text near the last point
        text_rect = text_item.boundingRect()
        text_item.setPos(
            last_point.x() + 10,
            last_point.y() - text_rect.height()/2
        )
        text_item.setZValue(11)  # Above lines
        
        self.scene.addItem(text_item)
        items.append(text_item)
        
        # Tag items with measurement_id for selection and store in measurements
        measurement_id = self.next_measurement_id
        
        measurement_data = {
            'id': measurement_id,
            'type': 'path',
            'points': [(p.x(), p.y()) for p in self.current_points],
            'pixel_length': total_pixel_length,
            'real_length': real_length,
            'unit': self.measurement_unit
        }
        
        for item in items:
            item.measurement_id = measurement_id
        
        self.measurements[measurement_id] = measurement_data
        self.measurement_items[measurement_id] = items
        self.next_measurement_id += 1
        
        # Emit signal with measurement data
        self.measurement_created.emit(measurement_data)

    def create_area_measurement(self):
        """Creates a persistent area measurement on the canvas."""
        if len(self.current_points) < 3 or not self.pixels_per_unit:
            return
        
        style = self.measurement_styles["area"]
        
        # Create a group of items for this measurement
        items = []
        
        # Create the polygon
        path = QPainterPath()
        path.moveTo(self.current_points[0])
        
        for i in range(1, len(self.current_points)):
            path.lineTo(self.current_points[i])
        
        path.closeSubpath()
        
        polygon_item = QGraphicsPathItem(path)
        polygon_item.setPen(style["pen"])
        polygon_item.setBrush(style["fill_brush"])
        polygon_item.setZValue(10)  # Above image
        self.scene.addItem(polygon_item)
        items.append(polygon_item)
        
        # Create points for vertices
        handle_size = style["handle_size"]
        
        for point in self.current_points:
            point_item = QGraphicsEllipseItem(
                point.x() - handle_size/2, point.y() - handle_size/2,
                handle_size, handle_size
            )
            point_item.setPen(QPen(style["pen"].color()))
            point_item.setBrush(QBrush(Qt.white))
            point_item.setZValue(11)  # Above polygon
            self.scene.addItem(point_item)
            items.append(point_item)
        
        # Calculate the area
        pixel_area = self.calculate_polygon_area(self.current_points)
        real_area = pixel_area / (self.pixels_per_unit ** 2)
        
        # Calculate centroid for text placement
        centroid_x = sum(p.x() for p in self.current_points) / len(self.current_points)
        centroid_y = sum(p.y() for p in self.current_points) / len(self.current_points)
        
        # Create label with area
        text_item = QGraphicsTextItem(f"Area: {real_area:.2f} {self.measurement_unit}²")
        text_item.setDefaultTextColor(style["text_color"])
        text_item.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Position text at centroid
        text_rect = text_item.boundingRect()
        text_item.setPos(
            centroid_x - text_rect.width()/2,
            centroid_y - text_rect.height()/2
        )
        text_item.setZValue(11)  # Above polygon
        
        self.scene.addItem(text_item)
        items.append(text_item)
        
        # Tag items with measurement_id for selection and store in measurements
        measurement_id = self.next_measurement_id
        
        measurement_data = {
            'id': measurement_id,
            'type': 'area',
            'points': [(p.x(), p.y()) for p in self.current_points],
            'pixel_area': pixel_area,
            'real_area': real_area,
            'unit': self.measurement_unit
        }
        
        for item in items:
            item.measurement_id = measurement_id
        
        self.measurements[measurement_id] = measurement_data
        self.measurement_items[measurement_id] = items
        self.next_measurement_id += 1
        
        # Emit signal with measurement data
        self.measurement_created.emit(measurement_data)

    def create_annotation(self, pos: QPointF, text: str):
        """
        Creates a persistent annotation on the canvas.
        
        Args:
            pos: Position for the annotation
            text: Annotation text
        """
        style = self.measurement_styles["annotation"]
        
        # Create a group of items for this annotation
        items = []
        
        # Create a marker point
        handle_size = style["handle_size"]
        
        marker = QGraphicsEllipseItem(
            pos.x() - handle_size/2, pos.y() - handle_size/2,
            handle_size, handle_size
        )
        marker.setPen(style["pen"])
        marker.setBrush(QBrush(Qt.white))
        marker.setZValue(11)
        self.scene.addItem(marker)
        items.append(marker)
        
        # Create text item
        text_item = QGraphicsTextItem(text)
        text_item.setDefaultTextColor(style["text_color"])
        text_item.setFont(QFont("Arial", 10))
        
        # Position text to the right of the marker
        text_item.setPos(pos.x() + handle_size, pos.y() - text_item.boundingRect().height()/2)
        text_item.setZValue(11)
        
        self.scene.addItem(text_item)
        items.append(text_item)
        
        # Tag items with measurement_id for selection and store in measurements
        measurement_id = self.next_measurement_id
        
        measurement_data = {
            'id': measurement_id,
            'type': 'annotation',
            'point': (pos.x(), pos.y()),
            'text': text
        }
        
        for item in items:
            item.measurement_id = measurement_id
        
        self.measurements[measurement_id] = measurement_data
        self.measurement_items[measurement_id] = items
        self.next_measurement_id += 1
        
        # Emit signal with measurement data
        self.measurement_created.emit(measurement_data)

    def calculate_polygon_area(self, points: List[QPointF]) -> float:
        """
        Calculates the area of a polygon using the Shoelace formula.
        
        Args:
            points: List of polygon vertices
            
        Returns:
            Area of the polygon in square pixels
        """
        n = len(points)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x() * points[j].y()
            area -= points[j].x() * points[i].y()
        
        area = abs(area) / 2.0
        return area

    def clear_temp_items(self):
        """Clears temporary visualization items."""
        for item in self.temp_items:
            self.scene.removeItem(item)
        self.temp_items = []

    def delete_measurement(self, measurement_id: int):
        """
        Deletes a measurement from the canvas.
        
        Args:
            measurement_id: ID of the measurement to delete
        """
        if measurement_id in self.measurement_items:
            # Remove graphics items
            for item in self.measurement_items[measurement_id]:
                self.scene.removeItem(item)
            
            # Remove from collections
            del self.measurement_items[measurement_id]
            del self.measurements[measurement_id]
            
            # Reset selected measurement if we just deleted it
            if self.selected_measurement_id == measurement_id:
                self.selected_measurement_id = None
            
            # Emit signal that measurement was deleted
            self.measurement_deleted.emit(measurement_id)

    def delete_all_measurements(self):
        """Deletes all measurements from the canvas."""
        for measurement_id in list(self.measurement_items.keys()):
            self.delete_measurement(measurement_id)

    def update_measurement_appearance(self):
        """Updates the appearance of measurements based on selection state."""
        if not self.highlight_selected or self.selected_measurement_id is None:
            return
        
        for measurement_id, items in self.measurement_items.items():
            # Check if this measurement is selected
            is_selected = (measurement_id == self.selected_measurement_id)
            
            for item in items:
                if isinstance(item, QGraphicsLineItem) or isinstance(item, QGraphicsPathItem):
                    # Adjust the width of lines and paths
                    pen = item.pen()
                    if is_selected:
                        pen.setWidth(pen.width() + 2)  # Make selected items thicker
                    else:
                        # Reset to normal width
                        style_type = self.measurements[measurement_id]['type']
                        style = self.measurement_styles[style_type]
                        pen.setWidth(style["pen"].width())
                    item.setPen(pen)
                elif isinstance(item, QGraphicsTextItem):
                    # Make text bold if selected
                    font = item.font()
                    font.setBold(is_selected)
                    item.setFont(font)

    def update_all_measurement_labels(self):
        """Updates all measurement labels with current calibration."""
        if not self.pixels_per_unit:
            return
            
        for measurement_id, data in self.measurements.items():
            if data['type'] == 'length' or data['type'] == 'path':
                # Recalculate real length based on current calibration
                points = data['points']
                if data['type'] == 'length' and len(points) == 2:
                    # Calculate length for a simple length measurement
                    p1 = QPointF(points[0][0], points[0][1])
                    p2 = QPointF(points[1][0], points[1][1])
                    pixel_length = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                    real_length = pixel_length / self.pixels_per_unit
                    
                    # Update data
                    data['pixel_length'] = pixel_length
                    data['real_length'] = real_length
                    data['unit'] = self.measurement_unit
                    
                elif data['type'] == 'path' and len(points) >= 2:
                    # Calculate length for a path measurement
                    total_pixel_length = 0
                    
                    for i in range(1, len(points)):
                        p1 = QPointF(points[i-1][0], points[i-1][1])
                        p2 = QPointF(points[i][0], points[i][1])
                        segment_length = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                        total_pixel_length += segment_length
                    
                    real_length = total_pixel_length / self.pixels_per_unit
                    
                    # Update data
                    data['pixel_length'] = total_pixel_length
                    data['real_length'] = real_length
                    data['unit'] = self.measurement_unit
                
                # Update the text label
                for item in self.measurement_items[measurement_id]:
                    if isinstance(item, QGraphicsTextItem):
                        if data['type'] == 'length':
                            item.setPlainText(f"{data['real_length']:.2f} {self.measurement_unit}")
                        else:  # path
                            item.setPlainText(f"Path: {data['real_length']:.2f} {self.measurement_unit}")
                
            elif data['type'] == 'area':
                # Recalculate area based on current calibration
                points = [QPointF(p[0], p[1]) for p in data['points']]
                pixel_area = self.calculate_polygon_area(points)
                real_area = pixel_area / (self.pixels_per_unit ** 2)
                
                # Update data
                data['pixel_area'] = pixel_area
                data['real_area'] = real_area
                data['unit'] = self.measurement_unit
                
                # Update the text label
                for item in self.measurement_items[measurement_id]:
                    if isinstance(item, QGraphicsTextItem):
                        item.setPlainText(f"Area: {real_area:.2f} {self.measurement_unit}²")

    def mouseMoveEvent(self, event):
        """
        Handles mouse move events for interactive measurements and drawing.
        
        Args:
            event: Mouse event
        """
        if not self.image_item.pixmap():
            super().mouseMoveEvent(event)
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        # In drawing modes, update visualization by replacing the last point
        if self.measurement_mode != MeasurementMode.VIEWING and len(self.current_points) > 0:
            # Snap to edge if enabled
            if self.enable_edge_snapping:
                scene_pos = self.snap_to_edge(scene_pos)
                
            # For in-progress measurements, update the last point
            if self.measurement_mode == MeasurementMode.LENGTH_MEASUREMENT and len(self.current_points) == 1:
                self.current_points.append(scene_pos)
                self.update_length_measurement_visualization()
                self.current_points.pop()  # Remove the temporary point
            elif self.measurement_mode == MeasurementMode.PATH_MEASUREMENT and len(self.current_points) > 0:
                self.current_points.append(scene_pos)
                self.update_path_measurement_visualization()
                self.current_points.pop()  # Remove the temporary point
            elif self.measurement_mode == MeasurementMode.AREA_MEASUREMENT and len(self.current_points) > 0:
                self.current_points.append(scene_pos)
                self.update_area_measurement_visualization()
                self.current_points.pop()  # Remove the temporary point
            elif self.measurement_mode == MeasurementMode.CALIBRATION and len(self.current_points) == 1:
                self.current_points.append(scene_pos)
                self.update_calibration_visualization()
                self.current_points.pop()  # Remove the temporary point
        
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """
        Handles mouse wheel events for zooming.
        
        Args:
            event: Wheel event
        """
        zoom_factor = 1.15
        
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def keyPressEvent(self, event):
        """
        Handles key press events for shortcuts and commands.
        
        Args:
            event: Key event
        """
        # Delete key to remove selected measurement
        if event.key() == Qt.Key_Delete and self.selected_measurement_id is not None:
            self.delete_measurement(self.selected_measurement_id)
            event.accept()
            return
            
        # Escape key to cancel current measurement
        if event.key() == Qt.Key_Escape:
            if len(self.current_points) > 0:
                self.clear_temp_items()
                self.current_points = []
                event.accept()
                return
                
        super().keyPressEvent(event)

    def show_context_menu(self, position):
        """
        Shows a context menu with measurement options.
        
        Args:
            position: Position for the menu in view coordinates
        """
        scene_pos = self.mapToScene(position)
        
        # Create menu
        menu = QMenu(self)
        
        # Check if we're over an existing measurement
        item = self.itemAt(position)
        if item and hasattr(item, 'measurement_id'):
            measurement_id = item.measurement_id
            
            # Actions for an existing measurement
            delete_action = QAction("Delete Measurement", self)
            delete_action.triggered.connect(lambda: self.delete_measurement(measurement_id))
            menu.addAction(delete_action)
            
            menu.addSeparator()
        
        # Actions for measurement creation
        if self.measurement_mode != MeasurementMode.VIEWING:
            cancel_action = QAction("Cancel Measurement", self)
            cancel_action.triggered.connect(self.cancel_current_measurement)
            menu.addAction(cancel_action)
        
        menu.addSeparator()
        
        # Mode selection actions
        view_action = QAction("View Mode", self)
        view_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.VIEWING))
        menu.addAction(view_action)
        
        length_action = QAction("Length Measurement", self)
        length_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.LENGTH_MEASUREMENT))
        menu.addAction(length_action)
        
        path_action = QAction("Path Measurement", self)
        path_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.PATH_MEASUREMENT))
        menu.addAction(path_action)
        
        area_action = QAction("Area Measurement", self)
        area_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.AREA_MEASUREMENT))
        menu.addAction(area_action)
        
        calibration_action = QAction("Calibration", self)
        calibration_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.CALIBRATION))
        menu.addAction(calibration_action)
        
        annotation_action = QAction("Add Annotation", self)
        annotation_action.triggered.connect(lambda: self.set_measurement_mode(MeasurementMode.ANNOTATION))
        menu.addAction(annotation_action)
        
        menu.addSeparator()
        
        # Toggle actions
        snapping_action = QAction("Enable Edge Snapping", self)
        snapping_action.setCheckable(True)
        snapping_action.setChecked(self.enable_edge_snapping)
        snapping_action.triggered.connect(lambda checked: setattr(self, 'enable_edge_snapping', checked))
        menu.addAction(snapping_action)
        
        grid_action = QAction("Show Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(self.show_grid)
        grid_action.triggered.connect(self.toggle_grid)
        menu.addAction(grid_action)
        
        menu.addSeparator()
        
        # Delete all measurements
        clear_action = QAction("Clear All Measurements", self)
        clear_action.triggered.connect(self.delete_all_measurements)
        menu.addAction(clear_action)
        
        # Show the menu
        menu.exec_(self.mapToGlobal(position))

    def cancel_current_measurement(self):
        """Cancels the current in-progress measurement."""
        self.clear_temp_items()
        self.current_points = []
        self.set_measurement_mode(MeasurementMode.VIEWING)

    def toggle_grid(self, show: bool):
        """
        Toggles the display of a measurement grid.
        
        Args:
            show: Whether to show the grid
        """
        self.show_grid = show
        
        # Remove existing grid if any
        for item in self.scene.items():
            if hasattr(item, 'is_grid_item') and item.is_grid_item:
                self.scene.removeItem(item)
        
        if show and self.image_item.pixmap():
            # Create a new grid
            rect = self.image_item.pixmap().rect()
            
            # Create horizontal grid lines
            for y in range(0, rect.height(), self.grid_spacing):
                line = QGraphicsLineItem(rect.left(), y, rect.right(), y)
                line.setPen(QPen(QColor(100, 100, 100, 100), 0.5, Qt.DashLine))
                line.setZValue(5)  # Above image, below measurements
                line.is_grid_item = True
                self.scene.addItem(line)
            
            # Create vertical grid lines
            for x in range(0, rect.width(), self.grid_spacing):
                line = QGraphicsLineItem(x, rect.top(), x, rect.bottom())
                line.setPen(QPen(QColor(100, 100, 100, 100), 0.5, Qt.DashLine))
                line.setZValue(5)  # Above image, below measurements
                line.is_grid_item = True
                self.scene.addItem(line)

    def get_all_measurements(self) -> Dict[int, Dict]:
        """
        Gets all measurements on the canvas.
        
        Returns:
            Dictionary mapping measurement IDs to measurement data
        """
        return self.measurements.copy()

    def get_measurement(self, measurement_id: int) -> Optional[Dict]:
        """
        Gets data for a specific measurement.
        
        Args:
            measurement_id: ID of the measurement
            
        Returns:
            Measurement data or None if not found
        """
        return self.measurements.get(measurement_id)

    def export_measurements_as_dict(self) -> Dict:
        """
        Exports all measurements in a serializable dictionary format.
        
        Returns:
            Dictionary of all measurements and calibration information
        """
        return {
            'calibration': {
                'pixels_per_unit': self.pixels_per_unit,
                'unit': self.measurement_unit
            },
            'measurements': self.measurements
        }
```

## ui/customer/customer_management_tab.py
```python
"""
Customer Management Tab for the main application window.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTableView, QAbstractItemView, QLineEdit, QComboBox, 
                           QDialog, QFormLayout, QMessageBox, QTextEdit, QFileDialog,
                           QMenu, QAction, QHeaderView, QSplitter, QTabWidget)
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QColor
from ui.ui_factory import UIFactory
from utils.notification_manager import NotificationManager
from utils.error_handling import ErrorHandler
from data.repositories.customer_repository import CustomerRepository
from models.customer import Customer, PropertyDetails, CustomerHistory, PrivacyControls
from typing import List, Dict, Any, Optional
import csv
import json
import asyncio
from datetime import datetime
import re

class CustomerTableModel(QAbstractTableModel):
    """Custom table model for customer data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.customers = []
        self.headers = ["ID", "Name", "Phone", "Email", "Type", "Last Updated"]
        
    def rowCount(self, parent=QModelIndex()):
        return len(self.customers)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.customers)):
            return QVariant()
            
        customer = self.customers[index.row()]
        column = index.column()
        
        if role == Qt.DisplayRole:
            # Format data for display
            if column == 0:  # ID
                return str(customer.customer_id or "New")
            elif column == 1:  # Name
                return customer.name
            elif column == 2:  # Phone
                return customer.phone
            elif column == 3:  # Email
                return customer.email or "—"
            elif column == 4:  # Type
                return customer.customer_type
            elif column == 5:  # Last Updated
                return customer.last_updated.strftime("%Y-%m-%d %H:%M")
        
        elif role == Qt.BackgroundRole:
            if customer.customer_id is None:  # New customer
                return QColor(240, 255, 240)  # Light green for new customers
                
        elif role == Qt.TextAlignmentRole:
            if column == 0:  # ID
                return Qt.AlignCenter
                
        elif role == Qt.UserRole:  # For storing the complete customer object
            return customer
            
        return QVariant()
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return QVariant()
        
    def setCustomers(self, customers: List[Customer]):
        """Set the customers list and refresh the model."""
        self.beginResetModel()
        self.customers = customers
        self.endResetModel()
        
    def addCustomer(self, customer: Customer):
        """Add a new customer to the model."""
        self.beginInsertRows(QModelIndex(), len(self.customers), len(self.customers))
        self.customers.append(customer)
        self.endInsertRows()
        
    def updateCustomer(self, index: int, customer: Customer):
        """Update a customer at the specified index."""
        if 0 <= index < len(self.customers):
            self.customers[index] = customer
            self.dataChanged.emit(
                self.index(index, 0),
                self.index(index, self.columnCount() - 1)
            )
            
    def removeCustomer(self, index: int):
        """Remove a customer at the specified index."""
        if 0 <= index < len(self.customers):
            self.beginRemoveRows(QModelIndex(), index, index)
            self.customers.pop(index)
            self.endRemoveRows()

class CustomerDialog(QDialog):
    """Dialog for creating or editing a customer."""
    
    def __init__(self, parent=None, customer: Optional[Customer] = None):
        super().__init__(parent)
        self.customer = customer or Customer(
            name="",
            email=None,
            phone="",
            secondary_phone=None,
            billing_address="",
            property_details=PropertyDetails(address=""),
            history=CustomerHistory(),
            privacy_controls=PrivacyControls(),
            customer_type="Residential",
            notes=None
        )
        self.setWindowTitle("Edit Customer" if customer else "New Customer")
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different sections
        tabs = QTabWidget()
        
        # Basic Information Tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        
        self.name_edit = QLineEdit(self.customer.name)
        self.name_edit.setPlaceholderText("Customer name")
        basic_layout.addRow("Name:", self.name_edit)
        
        self.email_edit = QLineEdit(self.customer.email or "")
        self.email_edit.setPlaceholderText("Email address")
        basic_layout.addRow("Email:", self.email_edit)
        
        self.phone_edit = QLineEdit(self.customer.phone)
        self.phone_edit.setPlaceholderText("Primary phone number")
        basic_layout.addRow("Phone:", self.phone_edit)
        
        self.sec_phone_edit = QLineEdit(self.customer.secondary_phone or "")
        self.sec_phone_edit.setPlaceholderText("Secondary phone (optional)")
        basic_layout.addRow("Secondary Phone:", self.sec_phone_edit)
        
        self.billing_address_edit = QTextEdit()
        self.billing_address_edit.setPlainText(self.customer.billing_address)
        self.billing_address_edit.setMaximumHeight(80)
        basic_layout.addRow("Billing Address:", self.billing_address_edit)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Residential", "Commercial", "Industrial"])
        self.type_combo.setCurrentText(self.customer.customer_type)
        basic_layout.addRow("Customer Type:", self.type_combo)
        
        tabs.addTab(basic_tab, "Basic Information")
        
        # Property Details Tab
        property_tab = QWidget()
        property_layout = QFormLayout(property_tab)
        
        self.property_address_edit = QTextEdit()
        self.property_address_edit.setPlainText(self.customer.property_details.address)
        self.property_address_edit.setMaximumHeight(80)
        property_layout.addRow("Property Address:", self.property_address_edit)
        
        self.property_type_edit = QLineEdit(self.customer.property_details.property_type or "")
        self.property_type_edit.setPlaceholderText("e.g., Single Family, Apartment, Commercial Building")
        property_layout.addRow("Property Type:", self.property_type_edit)
        
        self.building_age_edit = QLineEdit(str(self.customer.property_details.building_age or ""))
        self.building_age_edit.setPlaceholderText("Age in years")
        property_layout.addRow("Building Age:", self.building_age_edit)
        
        self.panel_info_edit = QLineEdit(self.customer.property_details.electrical_panel_info or "")
        self.panel_info_edit.setPlaceholderText("e.g., 200A, Federal Pacific")
        property_layout.addRow("Panel Info:", self.panel_info_edit)
        
        self.wiring_type_edit = QLineEdit(self.customer.property_details.wiring_type or "")
        self.wiring_type_edit.setPlaceholderText("e.g., Romex, Knob & Tube")
        property_layout.addRow("Wiring Type:", self.wiring_type_edit)
        
        self.property_notes_edit = QTextEdit()
        self.property_notes_edit.setPlainText(self.customer.property_details.notes or "")
        property_layout.addRow("Property Notes:", self.property_notes_edit)
        
        tabs.addTab(property_tab, "Property Details")
        
        # Notes Tab
        notes_tab = QWidget()
        notes_layout = QVBoxLayout(notes_tab)
        
        notes_label = QLabel("Customer Notes:")
        notes_layout.addWidget(notes_label)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.customer.notes or "")
        notes_layout.addWidget(self.notes_edit)
        
        tabs.addTab(notes_tab, "Notes")
        
        # Privacy Tab
        privacy_tab = QWidget()
        privacy_layout = QFormLayout(privacy_tab)
        
        from PyQt5.QtWidgets import QCheckBox
        
        self.contact_consent_check = QCheckBox()
        self.contact_consent_check.setChecked(self.customer.privacy_controls.consent_to_contact)
        privacy_layout.addRow("Consent to Contact:", self.contact_consent_check)
        
        self.marketing_consent_check = QCheckBox()
        self.marketing_consent_check.setChecked(self.customer.privacy_controls.consent_to_email_marketing)
        privacy_layout.addRow("Consent to Marketing:", self.marketing_consent_check)
        
        self.data_sharing_edit = QLineEdit(self.customer.privacy_controls.data_sharing_preferences or "")
        self.data_sharing_edit.setPlaceholderText("e.g., No sharing with third parties")
        privacy_layout.addRow("Data Sharing Preferences:", self.data_sharing_edit)
        
        self.privacy_notes_edit = QTextEdit()
        self.privacy_notes_edit.setPlainText(self.customer.privacy_controls.privacy_notes or "")
        self.privacy_notes_edit.setMaximumHeight(80)
        privacy_layout.addRow("Privacy Notes:", self.privacy_notes_edit)
        
        tabs.addTab(privacy_tab, "Privacy")
        
        layout.addWidget(tabs)
        
        # Button row
        button_layout = QHBoxLayout()
        self.save_button = UIFactory.create_button("Save", self.accept)
        self.cancel_button = UIFactory.create_button("Cancel", self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def get_customer_data(self) -> Customer:
        """Get the customer data from the form fields."""
        try:
            # Try to convert building age to int if provided
            building_age = None
            if self.building_age_edit.text().strip():
                try:
                    building_age = int(self.building_age_edit.text())
                except ValueError:
                    pass  # Leave as None if not a valid integer
            
            # Create updated Customer object
            updated_customer = Customer(
                name=self.name_edit.text(),
                email=self.email_edit.text() or None,
                phone=self.phone_edit.text(),
                secondary_phone=self.sec_phone_edit.text() or None,
                billing_address=self.billing_address_edit.toPlainText(),
                property_details=PropertyDetails(
                    address=self.property_address_edit.toPlainText(),
                    property_type=self.property_type_edit.text() or None,
                    building_age=building_age,
                    electrical_panel_info=self.panel_info_edit.text() or None,
                    wiring_type=self.wiring_type_edit.text() or None,
                    notes=self.property_notes_edit.toPlainText() or None
                ),
                history=self.customer.history,  # Keep existing history
                privacy_controls=PrivacyControls(
                    consent_to_contact=self.contact_consent_check.isChecked(),
                    consent_to_email_marketing=self.marketing_consent_check.isChecked(),
                    data_sharing_preferences=self.data_sharing_edit.text() or None,
                    privacy_notes=self.privacy_notes_edit.toPlainText() or None
                ),
                customer_type=self.type_combo.currentText(),
                notes=self.notes_edit.toPlainText() or None,
                customer_id=self.customer.customer_id,  # Keep existing ID
                date_added=self.customer.date_added,  # Keep original creation date
                last_updated=datetime.now()  # Update the last_updated time
            )
            
            return updated_customer
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "CustomerDialog.get_customer_data"})
            raise

class CustomerManagementTab(QWidget):
    """
    Customer Management Tab - displays and manages customer information.
    """
    
    customer_selected = pyqtSignal(Customer)  # Signal emitted when a customer is selected

    def __init__(self, notification_manager: NotificationManager, customer_repository: CustomerRepository):
        """
        Initializes the CustomerManagementTab.

        Args:
            notification_manager: NotificationManager instance for notifications.
            customer_repository: CustomerRepository for data access.
        """
        super().__init__()
        self.notification_manager = notification_manager
        self.customer_repository = customer_repository
        self.setup_ui()
        
        # Track currently selected customer
        self.selected_customer = None
        
        # Connect signals
        self.search_bar.textChanged.connect(self.filter_customers)
        self.filter_type_combo.currentIndexChanged.connect(self.filter_customers)
        self.customers_table_view.doubleClicked.connect(self.edit_customer)
        
        # Async task management
        self.tasks = []
        
        # Load initial data
        self.load_customer_data()

    def setup_ui(self):
        """Sets up the user interface for the customer management tab."""
        main_layout = QVBoxLayout(self)

        # Controls Bar (Buttons and Filters)
        controls_layout = QHBoxLayout()
        
        # Create button group for customer actions
        self.create_button = UIFactory.create_button("Create Customer", self.create_customer)
        self.create_button.setIcon(QIcon("icons/add_customer.png"))
        
        self.edit_button = UIFactory.create_button("Edit Customer", self.edit_customer)
        self.edit_button.setIcon(QIcon("icons/edit_customer.png"))
        self.edit_button.setEnabled(False)  # Disabled until customer is selected
        
        self.delete_button = UIFactory.create_button("Delete Customer", self.delete_customer)
        self.delete_button.setIcon(QIcon("icons/delete_customer.png"))
        self.delete_button.setEnabled(False)  # Disabled until customer is selected
        
        controls_layout.addWidget(self.create_button)
        controls_layout.addWidget(self.edit_button)
        controls_layout.addWidget(self.delete_button)
        
        # Add separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #888;")
        controls_layout.addWidget(separator)
        
        # Import/Export buttons
        self.import_button = UIFactory.create_button("Import", self.import_customers)
        self.import_button.setIcon(QIcon("icons/import.png"))
        self.export_button = UIFactory.create_button("Export", self.export_customers)
        self.export_button.setIcon(QIcon("icons/export.png"))
        
        controls_layout.addWidget(self.import_button)
        controls_layout.addWidget(self.export_button)
        
        controls_layout.addStretch()
        
        # Filter and Search
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["All Types", "Residential", "Commercial", "Industrial"])
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search Customers...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.setMinimumWidth(200)
        
        controls_layout.addWidget(QLabel("Type:"))
        controls_layout.addWidget(self.filter_type_combo)
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_bar)
        
        main_layout.addLayout(controls_layout)

        # Create a splitter for customer list and details
        splitter = QSplitter(Qt.Vertical)
        
        # Customer Table
        self.customers_table_view = QTableView()
        self.customers_table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.customers_table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.customers_table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.customers_table_view.setSortingEnabled(True)
        self.customers_table_view.verticalHeader().setVisible(False)
        self.customers_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.customers_table_view.horizontalHeader().setStretchLastSection(True)
        
        # Set up customer model and proxy model
        self.customer_model = CustomerTableModel()
        self.sort_filter_proxy_model = QSortFilterProxyModel()
        self.sort_filter_proxy_model.setSourceModel(self.customer_model)
        self.sort_filter_proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.sort_filter_proxy_model.setFilterKeyColumn(-1)  # Filter on all columns
        self.customers_table_view.setModel(self.sort_filter_proxy_model)
        
        # Connect selection changes
        self.customers_table_view.selectionModel().selectionChanged.connect(self.on_customer_selection_changed)
        
        splitter.addWidget(self.customers_table_view)
        
        # Customer Details Section
        self.details_tabs = QTabWidget()
        
        # Overview Tab
        self.overview_tab = QWidget()
        overview_layout = QFormLayout(self.overview_tab)
        
        self.details_name_label = QLabel("")
        self.details_name_label.setFont(QFont("Arial", 12, QFont.Bold))
        overview_layout.addRow("Name:", self.details_name_label)
        
        self.details_contact_label = QLabel("")
        overview_layout.addRow("Contact:", self.details_contact_label)
        
        self.details_address_label = QLabel("")
        overview_layout.addRow("Address:", self.details_address_label)
        
        self.details_type_label = QLabel("")
        overview_layout.addRow("Customer Type:", self.details_type_label)
        
        self.details_tabs.addTab(self.overview_tab, "Overview")
        
        # Property Details Tab
        self.property_tab = QWidget()
        property_layout = QFormLayout(self.property_tab)
        
        self.details_property_address_label = QLabel("")
        property_layout.addRow("Property Address:", self.details_property_address_label)
        
        self.details_property_type_label = QLabel("")
        property_layout.addRow("Property Type:", self.details_property_type_label)
        
        self.details_building_age_label = QLabel("")
        property_layout.addRow("Building Age:", self.details_building_age_label)
        
        self.details_panel_info_label = QLabel("")
        property_layout.addRow("Panel Info:", self.details_panel_info_label)
        
        self.details_wiring_type_label = QLabel("")
        property_layout.addRow("Wiring Type:", self.details_wiring_type_label)
        
        self.details_property_notes_label = QLabel("")
        self.details_property_notes_label.setWordWrap(True)
        property_layout.addRow("Property Notes:", self.details_property_notes_label)
        
        self.details_tabs.addTab(self.property_tab, "Property Details")
        
        # History Tab
        self.history_tab = QWidget()
        history_layout = QVBoxLayout(self.history_tab)
        
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        history_layout.addWidget(self.history_list)
        
        self.details_tabs.addTab(self.history_tab, "History")
        
        # Notes Tab
        self.notes_tab = QWidget()
        notes_layout = QVBoxLayout(self.notes_tab)
        
        self.notes_display = QTextEdit()
        self.notes_display.setReadOnly(True)
        notes_layout.addWidget(self.notes_display)
        
        self.details_tabs.addTab(self.notes_tab, "Notes")
        
        splitter.addWidget(self.details_tabs)
        
        # Set initial sizes for splitter - 2/3 for table, 1/3 for details
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.count_label = QLabel("0 customers")
        status_layout.addWidget(self.count_label)
        
        main_layout.addLayout(status_layout)

    def on_customer_selection_changed(self, selected, deselected):
        """Handle customer selection changes in the table."""
        indexes = selected.indexes()
        
        if indexes:
            # Get the selected customer from the model
            proxy_index = indexes[0]
            source_index = self.sort_filter_proxy_model.mapToSource(proxy_index)
            customer = self.customer_model.data(source_index, Qt.UserRole)
            
            if customer:
                self.selected_customer = customer
                self.update_customer_details(customer)
                self.edit_button.setEnabled(True)
                self.delete_button.setEnabled(True)
                
                # Emit signal that a customer was selected
                self.customer_selected.emit(customer)
        else:
            self.selected_customer = None
            self.clear_customer_details()
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)

    def update_customer_details(self, customer: Customer):
        """Update the detail panels with the selected customer's information."""
        # Overview Tab
        self.details_name_label.setText(customer.name)
        
        contact_text = f"<b>Phone:</b> {customer.phone}"
        if customer.secondary_phone:
            contact_text += f"<br><b>Alt Phone:</b> {customer.secondary_phone}"
        if customer.email:
            contact_text += f"<br><b>Email:</b> {customer.email}"
        self.details_contact_label.setText(contact_text)
        
        self.details_address_label.setText(customer.billing_address.replace("\n", "<br>"))
        self.details_type_label.setText(customer.customer_type)
        
        # Property Details Tab
        self.details_property_address_label.setText(customer.property_details.address.replace("\n", "<br>"))
        self.details_property_type_label.setText(customer.property_details.property_type or "—")
        self.details_building_age_label.setText(str(customer.property_details.building_age or "—"))
        self.details_panel_info_label.setText(customer.property_details.electrical_panel_info or "—")
        self.details_wiring_type_label.setText(customer.property_details.wiring_type or "—")
        self.details_property_notes_label.setText(customer.property_details.notes or "—")
        
        # History Tab
        history_text = ""
        
        if customer.history.previous_estimates:
            history_text += "<h3>Previous Estimates</h3>"
            for estimate_id in customer.history.previous_estimates:
                history_text += f"• Estimate #{estimate_id}<br>"
            history_text += "<br>"
        
        if customer.history.previous_projects:
            history_text += "<h3>Previous Projects</h3>"
            for project_id in customer.history.previous_projects:
                history_text += f"• Project #{project_id}<br>"
            history_text += "<br>"
        
        if customer.history.interaction_history:
            history_text += "<h3>Interaction History</h3>"
            for interaction in sorted(
                customer.history.interaction_history, 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            ):
                timestamp = interaction.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass
                
                history_text += f"<b>{timestamp}</b> - <i>{interaction.get('type', '')}</i><br>"
                history_text += f"{interaction.get('description', '')}<br><br>"
        
        if not history_text:
            history_text = "No history available for this customer."
            
        self.history_list.setHtml(history_text)
        
        # Notes Tab
        if customer.notes:
            self.notes_display.setPlainText(customer.notes)
        else:
            self.notes_display.setPlainText("No notes available for this customer.")

    def clear_customer_details(self):
        """Clear the customer details panels."""
        self.details_name_label.setText("")
        self.details_contact_label.setText("")
        self.details_address_label.setText("")
        self.details_type_label.setText("")
        
        self.details_property_address_label.setText("")
        self.details_property_type_label.setText("")
        self.details_building_age_label.setText("")
        self.details_panel_info_label.setText("")
        self.details_wiring_type_label.setText("")
        self.details_property_notes_label.setText("")
        
        self.history_list.clear()
        self.notes_display.clear()

    async def load_customers_async(self):
        """Asynchronously load customers from the repository."""
        try:
            self.status_label.setText("Loading customers...")
            
            # Get filter settings
            filter_type = self.filter_type_combo.currentText()
            search_text = self.search_bar.text()
            
            # Prepare filters
            filters = {}
            if filter_type != "All Types":
                filters["customer_type"] = filter_type
            
            # Get customers from repository
            customers = await self.customer_repository.list_customers(
                limit=1000,  # Reasonable limit for now - implement pagination for large datasets
                offset=0,
                filters=filters,
                search_term=search_text if search_text else None
            )
            
            # Get count for status bar
            count = await self.customer_repository.count_customers(
                filters=filters,
                search_term=search_text if search_text else None
            )
            
            # Update the UI with the retrieved customers
            self.customer_model.setCustomers(customers)
            self.count_label.setText(f"{count} customer{'' if count == 1 else 's'}")
            self.status_label.setText("Ready")
            
            # Reset column widths for better display
            self.customers_table_view.setColumnWidth(0, 70)  # ID
            self.customers_table_view.setColumnWidth(1, 200)  # Name
            self.customers_table_view.setColumnWidth(2, 120)  # Phone
            self.customers_table_view.setColumnWidth(3, 200)  # Email
            self.customers_table_view.setColumnWidth(4, 100)  # Type
            
            return customers
            
        except Exception as e:
            self.status_label.setText("Error loading customers")
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab.load_customers_async"})
            return []

    def load_customer_data(self):
        """Load customer data into the table view."""
        # Start asynchronous loading
        task = asyncio.create_task(self.load_customers_async())
        self.tasks.append(task)
        
        # Use task.add_done_callback to handle completion
        task.add_done_callback(self._load_complete_callback)

    def _load_complete_callback(self, task):
        """Callback when customer load is complete."""
        # Remove task from tracking
        if task in self.tasks:
            self.tasks.remove(task)
        
        # Check for exceptions
        if task.exception():
            ErrorHandler.handle_error(
                task.exception(), 
                {"context": "CustomerManagementTab._load_complete_callback"}
            )

    def refresh(self):
        """Refreshes the customer list."""
        self.load_customer_data()

    def filter_customers(self):
        """Filters customers based on search text and type."""
        # Reuse the load_customer_data method which already handles filters
        self.load_customer_data()

    def create_customer(self):
        """Create a new customer."""
        dialog = CustomerDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get customer data from dialog
                new_customer = dialog.get_customer_data()
                
                # Start async task to save customer
                task = asyncio.create_task(self._save_customer_async(new_customer))
                self.tasks.append(task)
                
                task.add_done_callback(lambda t: self._save_complete_callback(t, is_new=True))
                
            except Exception as e:
                ErrorHandler.handle_error(e, {"context": "CustomerManagementTab.create_customer"})

    def edit_customer(self):
        """Edit the selected customer."""
        if not self.selected_customer:
            self.notification_manager.notify(
                "Edit Customer", 
                "No customer selected", 
                level="warning"
            )
            return
            
        dialog = CustomerDialog(self, self.selected_customer)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get updated customer data from dialog
                updated_customer = dialog.get_customer_data()
                
                # Start async task to update customer
                task = asyncio.create_task(self._save_customer_async(updated_customer))
                self.tasks.append(task)
                
                task.add_done_callback(lambda t: self._save_complete_callback(t, is_new=False))
                
            except Exception as e:
                ErrorHandler.handle_error(e, {"context": "CustomerManagementTab.edit_customer"})

    async def _save_customer_async(self, customer: Customer) -> Customer:
        """Asynchronously save a customer (create or update)."""
        try:
            self.status_label.setText("Saving customer...")
            
            if customer.customer_id:
                # Update existing customer
                success = await self.customer_repository.update_customer(customer)
                if not success:
                    raise ValueError("Failed to update customer")
                return customer
            else:
                # Create new customer
                customer_id = await self.customer_repository.create_customer(customer)
                if not customer_id:
                    raise ValueError("Failed to create customer")
                
                # Update the customer object with the new ID
                customer.customer_id = customer_id
                return customer
                
        except Exception as e:
            self.status_label.setText("Error saving customer")
            raise e

    def _save_complete_callback(self, task, is_new: bool):
        """Callback when customer save is complete."""
        # Remove task from tracking
        if task in self.tasks:
            self.tasks.remove(task)
        
        try:
            # Get the result
            if task.exception():
                raise task.exception()
                
            customer = task.result()
            
            # Update UI
            if is_new:
                # Add to model
                self.customer_model.addCustomer(customer)
                self.notification_manager.notify(
                    "Customer Created", 
                    f"Created new customer: {customer.name}",
                    level="info"
                )
            else:
                # Find and update in model
                for i, existing_customer in enumerate(self.customer_model.customers):
                    if existing_customer.customer_id == customer.customer_id:
                        self.customer_model.updateCustomer(i, customer)
                        break
                
                # If this was the selected customer, update details
                if (self.selected_customer and 
                    self.selected_customer.customer_id == customer.customer_id):
                    self.selected_customer = customer
                    self.update_customer_details(customer)
                
                self.notification_manager.notify(
                    "Customer Updated", 
                    f"Updated customer: {customer.name}",
                    level="info"
                )
            
            self.status_label.setText("Customer saved successfully")
            
            # Refresh the view to show changes
            self.refresh()
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._save_complete_callback"})
            self.status_label.setText("Error saving customer")

    def delete_customer(self):
        """Delete the selected customer."""
        if not self.selected_customer:
            self.notification_manager.notify(
                "Delete Customer", 
                "No customer selected", 
                level="warning"
            )
            return
        
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete customer '{self.selected_customer.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Start async task to delete customer
            customer_id = self.selected_customer.customer_id
            task = asyncio.create_task(self._delete_customer_async(customer_id))
            self.tasks.append(task)
            
            task.add_done_callback(self._delete_complete_callback)

    async def _delete_customer_async(self, customer_id: int) -> bool:
        """Asynchronously delete a customer."""
        try:
            self.status_label.setText("Deleting customer...")
            
            # Delete the customer
            success = await self.customer_repository.delete_customer(customer_id)
            
            if not success:
                raise ValueError(f"Failed to delete customer ID: {customer_id}")
                
            return customer_id
            
        except Exception as e:
            self.status_label.setText("Error deleting customer")
            raise e

    def _delete_complete_callback(self, task):
        """Callback when customer delete is complete."""
        # Remove task from tracking
        if task in self.tasks:
            self.tasks.remove(task)
        
        try:
            # Get the result
            if task.exception():
                raise task.exception()
                
            customer_id = task.result()
            
            # Find and remove from model
            for i, customer in enumerate(self.customer_model.customers):
                if customer.customer_id == customer_id:
                    customer_name = customer.name
                    self.customer_model.removeCustomer(i)
                    
                    # Update status
                    self.notification_manager.notify(
                        "Customer Deleted", 
                        f"Deleted customer: {customer_name}",
                        level="info"
                    )
                    break
            
            # Clear the selected customer if it was deleted
            if (self.selected_customer and 
                self.selected_customer.customer_id == customer_id):
                self.selected_customer = None
                self.clear_customer_details()
                self.edit_button.setEnabled(False)
                self.delete_button.setEnabled(False)
            
            self.status_label.setText("Customer deleted successfully")
            
            # Refresh the view to show changes
            self.refresh()
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._delete_complete_callback"})
            self.status_label.setText("Error deleting customer")

    def import_customers(self):
        """Import customers from a file."""
        # Show file dialog to select import file
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Import Customers")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV Files (*.csv);;JSON Files (*.json)")
        
        if file_dialog.exec_() == QDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            
            # Determine import format based on file extension
            if file_path.lower().endswith('.csv'):
                self._import_from_csv(file_path)
            elif file_path.lower().endswith('.json'):
                self._import_from_json(file_path)
            else:
                self.notification_manager.notify(
                    "Import Error", 
                    "Unsupported file format. Please use CSV or JSON.",
                    level="error"
                )

    def _import_from_csv(self, file_path: str):
        """Import customers from a CSV file."""
        try:
            self.status_label.setText("Importing customers from CSV...")
            
            imported_customers = []
            
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                # Try to detect the CSV dialect
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(csvfile, dialect=dialect)
                
                # Map CSV fields to Customer model
                for row in reader:
                    try:
                        # Basic required fields
                        if not row.get('name') or not row.get('phone'):
                            continue  # Skip rows missing required fields
                        
                        # Create a new Customer object
                        customer = Customer(
                            name=row.get('name', ''),
                            email=row.get('email'),
                            phone=row.get('phone', ''),
                            secondary_phone=row.get('secondary_phone'),
                            billing_address=row.get('billing_address', ''),
                            property_details=PropertyDetails(
                                address=row.get('property_address', ''),
                                property_type=row.get('property_type'),
                                building_age=int(row['building_age']) if row.get('building_age') and row['building_age'].isdigit() else None,
                                electrical_panel_info=row.get('panel_info'),
                                wiring_type=row.get('wiring_type'),
                                notes=row.get('property_notes')
                            ),
                            history=CustomerHistory(),
                            privacy_controls=PrivacyControls(
                                consent_to_contact=row.get('consent_to_contact', '').lower() in ['true', 'yes', '1'],
                                consent_to_email_marketing=row.get('consent_to_email_marketing', '').lower() in ['true', 'yes', '1'],
                                data_sharing_preferences=row.get('data_sharing_preferences'),
                                privacy_notes=row.get('privacy_notes')
                            ),
                            customer_type=row.get('customer_type', 'Residential'),
                            notes=row.get('notes')
                        )
                        
                        imported_customers.append(customer)
                    except Exception as e:
                        # Log the error but continue with other rows
                        logger.warning(f"Error importing customer from CSV row: {e}")
                        continue
            
            # Create async task to save all imported customers
            if imported_customers:
                task = asyncio.create_task(self._import_customers_async(imported_customers))
                self.tasks.append(task)
                
                task.add_done_callback(self._import_complete_callback)
            else:
                self.notification_manager.notify(
                    "Import Results", 
                    "No valid customers found in the CSV file.",
                    level="warning"
                )
                self.status_label.setText("Import completed - no valid customers found")
                
        except Exception as e:
            self.status_label.setText("Error importing customers")
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._import_from_csv"})

    def _import_from_json(self, file_path: str):
        """Import customers from a JSON file."""
        try:
            self.status_label.setText("Importing customers from JSON...")
            
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                
                imported_customers = []
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # List of customers
                    customer_dicts = data
                elif isinstance(data, dict) and 'customers' in data:
                    # Object with customers array
                    customer_dicts = data['customers']
                else:
                    # Single customer
                    customer_dicts = [data]
                
                for customer_dict in customer_dicts:
                    try:
                        # Create customer from JSON
                        customer = Customer.parse_obj(customer_dict)
                        imported_customers.append(customer)
                    except Exception as e:
                        # Log the error but continue with other items
                        logger.warning(f"Error parsing customer from JSON: {e}")
                        continue
            
            # Create async task to save all imported customers
            if imported_customers:
                task = asyncio.create_task(self._import_customers_async(imported_customers))
                self.tasks.append(task)
                
                task.add_done_callback(self._import_complete_callback)
            else:
                self.notification_manager.notify(
                    "Import Results", 
                    "No valid customers found in the JSON file.",
                    level="warning"
                )
                self.status_label.setText("Import completed - no valid customers found")
                
        except Exception as e:
            self.status_label.setText("Error importing customers")
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._import_from_json"})

    async def _import_customers_async(self, customers: List[Customer]) -> Dict[str, int]:
        """Asynchronously import a list of customers."""
        try:
            results = {
                'success': 0,
                'error': 0
            }
            
            # Save each customer
            for customer in customers:
                try:
                    # Reset customer_id to ensure new customer is created
                    customer.customer_id = None
                    
                    # Create the customer
                    customer_id = await self.customer_repository.create_customer(customer)
                    
                    if customer_id:
                        results['success'] += 1
                    else:
                        results['error'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error saving imported customer: {e}")
                    results['error'] += 1
            
            return results
                
        except Exception as e:
            self.status_label.setText("Error importing customers")
            raise e

    def _import_complete_callback(self, task):
        """Callback when customer import is complete."""
        # Remove task from tracking
        if task in self.tasks:
            self.tasks.remove(task)
        
        try:
            # Get the result
            if task.exception():
                raise task.exception()
                
            results = task.result()
            
            # Show results notification
            self.notification_manager.notify(
                "Import Results", 
                f"Successfully imported {results['success']} customers. {results['error']} failed.",
                level="info" if results['success'] > 0 else "warning"
            )
            
            self.status_label.setText(f"Import completed: {results['success']} succeeded, {results['error']} failed")
            
            # Refresh the view to show imported customers
            self.refresh()
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._import_complete_callback"})
            self.status_label.setText("Error importing customers")

    def export_customers(self):
        """Export customers to a file."""
        # Create a menu with export options
        export_menu = QMenu(self)
        
        csv_action = QAction("Export as CSV", export_menu)
        csv_action.triggered.connect(lambda: self._export_as_csv())
        export_menu.addAction(csv_action)
        
        json_action = QAction("Export as JSON", export_menu)
        json_action.triggered.connect(lambda: self._export_as_json())
        export_menu.addAction(json_action)
        
        # Show filtered only checkbox
        filtered_action = QAction("Export Filtered Results Only", export_menu)
        filtered_action.setCheckable(True)
        filtered_action.setChecked(True)
        export_menu.addAction(filtered_action)
        
        # Get export options
        result = export_menu.exec_(QCursor.pos())
        
        # Check if an option was selected
        if not result:
            return
            
        # Get filtered setting
        export_filtered_only = filtered_action.isChecked()
        
        # Start export based on selected format
        if result == csv_action:
            self._export_as_csv(filtered=export_filtered_only)
        elif result == json_action:
            self._export_as_json(filtered=export_filtered_only)

    def _get_customers_for_export(self, filtered: bool) -> List[Customer]:
        """Get the customers to export based on filter settings."""
        if filtered:
            # Get visible customers from the proxy model
            customers = []
            for row in range(self.sort_filter_proxy_model.rowCount()):
                proxy_index = self.sort_filter_proxy_model.index(row, 0)
                source_index = self.sort_filter_proxy_model.mapToSource(proxy_index)
                customer = self.customer_model.data(source_index, Qt.UserRole)
                if customer:
                    customers.append(customer)
            return customers
        else:
            # Get all customers from the repository
            return self.customer_model.customers

    def _export_as_csv(self, filtered: bool = True):
        """Export customers as CSV."""
        # Show file dialog to select export file
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Export Customers as CSV")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("csv")
        file_dialog.setNameFilter("CSV Files (*.csv)")
        
        if file_dialog.exec_() == QDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                self.status_label.setText("Exporting customers to CSV...")
                
                # Get customers to export
                customers = self._get_customers_for_export(filtered)
                
                if not customers:
                    self.notification_manager.notify(
                        "Export Warning", 
                        "No customers to export.",
                        level="warning"
                    )
                    self.status_label.setText("Export completed - no customers to export")
                    return
                
                # Define CSV field names
                fieldnames = [
                    'name', 'email', 'phone', 'secondary_phone', 'billing_address',
                    'customer_type', 'property_address', 'property_type', 'building_age',
                    'panel_info', 'wiring_type', 'property_notes', 'notes',
                    'consent_to_contact', 'consent_to_email_marketing'
                ]
                
                # Write to CSV
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for customer in customers:
                        # Map customer data to CSV row
                        row = {
                            'name': customer.name,
                            'email': customer.email or '',
                            'phone': customer.phone,
                            'secondary_phone': customer.secondary_phone or '',
                            'billing_address': customer.billing_address,
                            'customer_type': customer.customer_type,
                            'property_address': customer.property_details.address,
                            'property_type': customer.property_details.property_type or '',
                            'building_age': customer.property_details.building_age or '',
                            'panel_info': customer.property_details.electrical_panel_info or '',
                            'wiring_type': customer.property_details.wiring_type or '',
                            'property_notes': customer.property_details.notes or '',
                            'notes': customer.notes or '',
                            'consent_to_contact': str(customer.privacy_controls.consent_to_contact),
                            'consent_to_email_marketing': str(customer.privacy_controls.consent_to_email_marketing)
                        }
                        
                        writer.writerow(row)
                
                self.notification_manager.notify(
                    "Export Completed", 
                    f"Successfully exported {len(customers)} customers to CSV.",
                    level="info"
                )
                self.status_label.setText(f"Export completed: {len(customers)} customers exported to CSV")
                
            except Exception as e:
                self.status_label.setText("Error exporting customers")
                ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._export_as_csv"})

    def _export_as_json(self, filtered: bool = True):
        """Export customers as JSON."""
        # Show file dialog to select export file
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Export Customers as JSON")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("json")
        file_dialog.setNameFilter("JSON Files (*.json)")
        
        if file_dialog.exec_() == QDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                self.status_label.setText("Exporting customers to JSON...")
                
                # Get customers to export
                customers = self._get_customers_for_export(filtered)
                
                if not customers:
                    self.notification_manager.notify(
                        "Export Warning", 
                        "No customers to export.",
                        level="warning"
                    )
                    self.status_label.setText("Export completed - no customers to export")
                    return
                
                # Convert customers to JSON-serializable format
                customer_dicts = []
                for customer in customers:
                    customer_dict = customer.dict(exclude={'history'})
                    customer_dicts.append(customer_dict)
                
                # Write to JSON file
                with open(file_path, 'w', encoding='utf-8') as jsonfile:
                    json.dump({
                        'customers': customer_dicts, 
                        'export_date': datetime.now().isoformat(),
                        'count': len(customer_dicts)
                    }, jsonfile, indent=2, default=str)
                
                self.notification_manager.notify(
                    "Export Completed", 
                    f"Successfully exported {len(customers)} customers to JSON.",
                    level="info"
                )
                self.status_label.setText(f"Export completed: {len(customers)} customers exported to JSON")
                
            except Exception as e:
                self.status_label.setText("Error exporting customers")
                ErrorHandler.handle_error(e, {"context": "CustomerManagementTab._export_as_json"})

    def add_interaction_to_customer(self, customer: Customer, interaction_type: str, description: str) -> bool:
        """
        Add an interaction to a customer's history.
        
        Args:
            customer: The customer to update
            interaction_type: Type of interaction
            description: Description of the interaction
            
        Returns:
            True if successful, False otherwise
        """
        if not customer or not customer.customer_id:
            return False
            
        try:
            # Add interaction to customer object
            customer.add_interaction_note(interaction_type, description)
            
            # Save customer asynchronously
            task = asyncio.create_task(self._save_customer_async(customer))
            self.tasks.append(task)
            
            # Use lambda to prevent _save_complete_callback from thinking this is a new customer
            task.add_done_callback(lambda t: self._save_complete_callback(t, is_new=False))
            
            return True
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "CustomerManagementTab.add_interaction_to_customer"})
            return False
```

## ui/scheduler/scheduler_tab.py
```python
"""
Scheduler Tab for the main application window, implementing task scheduling features.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QCalendarWidget, QListView, QDialog, QFormLayout, QLineEdit, 
                           QDateTimeEdit, QComboBox, QTextEdit, QMessageBox, QMenu,
                           QAction, QAbstractItemView, QTableView, QHeaderView)
from PyQt5.QtCore import Qt, QDate, QTime, QDateTime, QSortFilterProxyModel, pyqtSignal, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QColor, QBrush, QFont
from ui.ui_factory import UIFactory
from utils.notification_manager import NotificationManager
from utils.error_handling import ErrorHandler
from models.customer import Customer
from data.repositories.customer_repository import CustomerRepository
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional, Tuple

class TaskPriority:
    """Task priority levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3
    
    @staticmethod
    def to_string(priority: int) -> str:
        """Convert priority level to string."""
        if priority == TaskPriority.LOW:
            return "Low"
        elif priority == TaskPriority.MEDIUM:
            return "Medium"
        elif priority == TaskPriority.HIGH:
            return "High"
        elif priority == TaskPriority.URGENT:
            return "Urgent"
        return "Unknown"
    
    @staticmethod
    def to_color(priority: int) -> QColor:
        """Convert priority level to color."""
        if priority == TaskPriority.LOW:
            return QColor(200, 255, 200)  # Light green
        elif priority == TaskPriority.MEDIUM:
            return QColor(255, 255, 200)  # Light yellow
        elif priority == TaskPriority.HIGH:
            return QColor(255, 200, 200)  # Light red
        elif priority == TaskPriority.URGENT:
            return QColor(255, 150, 150)  # Darker red
        return QColor(255, 255, 255)  # White

class TaskStatus:
    """Task status types."""
    PLANNED = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    CANCELLED = 3
    
    @staticmethod
    def to_string(status: int) -> str:
        """Convert status to string."""
        if status == TaskStatus.PLANNED:
            return "Planned"
        elif status == TaskStatus.IN_PROGRESS:
            return "In Progress"
        elif status == TaskStatus.COMPLETED:
            return "Completed"
        elif status == TaskStatus.CANCELLED:
            return "Cancelled"
        return "Unknown"

class Task:
    """Represents a scheduled task."""
    
    def __init__(self, 
                title: str, 
                date_time: datetime, 
                description: str = "", 
                task_type: str = "General", 
                priority: int = TaskPriority.MEDIUM, 
                status: int = TaskStatus.PLANNED,
                customer_id: Optional[int] = None,
                assigned_to: Optional[str] = None,
                task_id: Optional[int] = None):
        """
        Initialize a new task.
        
        Args:
            title: Task title
            date_time: Date and time of the task
            description: Task description
            task_type: Type of task
            priority: Priority level
            status: Task status
            customer_id: Optional associated customer ID
            assigned_to: Optional person assigned to the task
            task_id: Optional task ID (None for new tasks)
        """
        self.task_id = task_id
        self.title = title
        self.date_time = date_time
        self.description = description
        self.task_type = task_type
        self.priority = priority
        self.status = status
        self.customer_id = customer_id
        self.assigned_to = assigned_to
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def is_overdue(self) -> bool:
        """Check if the task is overdue."""
        return (self.date_time < datetime.now() and 
                self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage."""
        return {
            'task_id': self.task_id,
            'title': self.title,
            'date_time': self.date_time.isoformat(),
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority,
            'status': self.status,
            'customer_id': self.customer_id,
            'assigned_to': self.assigned_to,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a task from a dictionary."""
        try:
            date_time = datetime.fromisoformat(data['date_time'])
            created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
            updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
            
            task = cls(
                title=data['title'],
                date_time=date_time,
                description=data.get('description', ''),
                task_type=data.get('task_type', 'General'),
                priority=data.get('priority', TaskPriority.MEDIUM),
                status=data.get('status', TaskStatus.PLANNED),
                customer_id=data.get('customer_id'),
                assigned_to=data.get('assigned_to'),
                task_id=data.get('task_id')
            )
            
            task.created_at = created_at
            task.updated_at = updated_at
            
            return task
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "Task.from_dict"})
            raise ValueError(f"Failed to create task from dictionary: {e}")

class TaskDialog(QDialog):
    """Dialog for creating or editing a task."""
    
    def __init__(self, parent=None, task: Optional[Task] = None, customers: List[Customer] = None):
        """
        Initialize the task dialog.
        
        Args:
            parent: Parent widget
            task: Task to edit or None for a new task
            customers: List of customers for customer selection
        """
        super().__init__(parent)
        self.task = task or Task(
            title="",
            date_time=datetime.now() + timedelta(hours=1),  # Default to 1 hour from now
            description="",
            task_type="General",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PLANNED
        )
        self.customers = customers or []
        self.setWindowTitle("Edit Task" if task else "New Task")
        self.setMinimumWidth(400)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # Task Title
        self.title_edit = QLineEdit(self.task.title)
        self.title_edit.setPlaceholderText("Task title")
        form_layout.addRow("Title:", self.title_edit)
        
        # Date and Time
        self.date_time_edit = QDateTimeEdit(QDateTime.fromDateTime(
            self.task.date_time.astimezone() if hasattr(self.task.date_time, 'astimezone') else self.task.date_time
        ))
        self.date_time_edit.setCalendarPopup(True)
        self.date_time_edit.setDisplayFormat("yyyy-MM-dd hh:mm AP")
        form_layout.addRow("Date & Time:", self.date_time_edit)
        
        # Task Type
        self.type_combo = QComboBox()
        task_types = ["General", "Estimate", "Installation", "Service Call", "Follow-up", "Inspection", "Meeting", "Other"]
        self.type_combo.addItems(task_types)
        current_index = task_types.index(self.task.task_type) if self.task.task_type in task_types else 0
        self.type_combo.setCurrentIndex(current_index)
        form_layout.addRow("Task Type:", self.type_combo)
        
        # Priority
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Low", "Medium", "High", "Urgent"])
        self.priority_combo.setCurrentIndex(self.task.priority)
        form_layout.addRow("Priority:", self.priority_combo)
        
        # Status
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Planned", "In Progress", "Completed", "Cancelled"])
        self.status_combo.setCurrentIndex(self.task.status)
        form_layout.addRow("Status:", self.status_combo)
        
        # Assigned To
        self.assigned_to_edit = QLineEdit(self.task.assigned_to or "")
        self.assigned_to_edit.setPlaceholderText("Person assigned to this task")
        form_layout.addRow("Assigned To:", self.assigned_to_edit)
        
        # Customer Selection
        self.customer_combo = QComboBox()
        self.customer_combo.addItem("None", None)  # No customer
        
        # Add customers to combo box
        selected_index = 0
        for i, customer in enumerate(self.customers):
            self.customer_combo.addItem(f"{customer.name} ({customer.phone})", customer.customer_id)
            if self.task.customer_id == customer.customer_id:
                selected_index = i + 1  # +1 because of the "None" item
        
        self.customer_combo.setCurrentIndex(selected_index)
        form_layout.addRow("Customer:", self.customer_combo)
        
        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setPlainText(self.task.description)
        self.description_edit.setPlaceholderText("Task description")
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Button row
        button_layout = QHBoxLayout()
        self.save_button = UIFactory.create_button("Save", self.accept)
        self.cancel_button = UIFactory.create_button("Cancel", self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def get_task_data(self) -> Task:
        """Get the task data from the form."""
        try:
            # Get date and time
            qdate_time = self.date_time_edit.dateTime()
            date_time = datetime(
                qdate_time.date().year(),
                qdate_time.date().month(),
                qdate_time.date().day(),
                qdate_time.time().hour(),
                qdate_time.time().minute()
            )
            
            # Get customer ID
            customer_id = self.customer_combo.currentData()
            
            # Update task object
            task = Task(
                title=self.title_edit.text(),
                date_time=date_time,
                description=self.description_edit.toPlainText(),
                task_type=self.type_combo.currentText(),
                priority=self.priority_combo.currentIndex(),
                status=self.status_combo.currentIndex(),
                customer_id=customer_id,
                assigned_to=self.assigned_to_edit.text() or None,
                task_id=self.task.task_id  # Keep existing ID
            )
            
            # Keep creation date if editing existing task
            if self.task.task_id is not None:
                task.created_at = self.task.created_at
            
            return task
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "TaskDialog.get_task_data"})
            raise

class TaskScheduler:
    """
    Task scheduling service for managing tasks.
    """
    
    def __init__(self):
        """Initialize the task scheduler."""
        self.tasks: Dict[int, Task] = {}
        self.next_task_id = 1
        
    def add_task(self, task: Task) -> int:
        """
        Add a new task.
        
        Args:
            task: The task to add
            
        Returns:
            The task ID
        """
        # Assign a new ID if needed
        if task.task_id is None:
            task.task_id = self.next_task_id
            self.next_task_id += 1
        
        # Update the task
        task.updated_at = datetime.now()
        
        # Store the task
        self.tasks[task.task_id] = task
        
        return task.task_id
    
    def update_task(self, task: Task) -> bool:
        """
        Update an existing task.
        
        Args:
            task: The task to update
            
        Returns:
            True if successful, False otherwise
        """
        if task.task_id is None or task.task_id not in self.tasks:
            return False
        
        # Update the task
        task.updated_at = datetime.now()
        
        # Store the task
        self.tasks[task.task_id] = task
        
        return True
    
    def delete_task(self, task_id: int) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: The ID of the task to delete
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        # Remove the task
        del self.tasks[task_id]
        
        return True
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: The ID of the task to get
            
        Returns:
            The task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_tasks_for_date(self, date: datetime.date) -> List[Task]:
        """
        Get tasks for a specific date.
        
        Args:
            date: The date to get tasks for
            
        Returns:
            List of tasks for the date
        """
        return [task for task in self.tasks.values() 
                if task.date_time.date() == date]
    
    def get_tasks_for_customer(self, customer_id: int) -> List[Task]:
        """
        Get tasks for a specific customer.
        
        Args:
            customer_id: The customer ID
            
        Returns:
            List of tasks for the customer
        """
        return [task for task in self.tasks.values() 
                if task.customer_id == customer_id]
    
    def get_all_tasks(self) -> List[Task]:
        """
        Get all tasks.
        
        Returns:
            List of all tasks
        """
        return list(self.tasks.values())
    
    def get_upcoming_tasks(self, days: int = 7) -> List[Task]:
        """
        Get upcoming tasks.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of upcoming tasks
        """
        now = datetime.now()
        end_date = now + timedelta(days=days)
        
        return [task for task in self.tasks.values() 
                if now <= task.date_time <= end_date]
    
    def get_overdue_tasks(self) -> List[Task]:
        """
        Get overdue tasks.
        
        Returns:
            List of overdue tasks
        """
        now = datetime.now()
        
        return [task for task in self.tasks.values() 
                if task.date_time < now and 
                task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]]
    
    def save_tasks_to_dict(self) -> Dict[str, Any]:
        """
        Save tasks to a dictionary for storage.
        
        Returns:
            Dictionary of task data
        """
        task_dicts = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        
        return {
            'tasks': task_dicts,
            'next_task_id': self.next_task_id
        }
    
    def load_tasks_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load tasks from a dictionary.
        
        Args:
            data: Dictionary of task data
        """
        if 'tasks' in data:
            self.tasks = {}
            for task_id_str, task_dict in data['tasks'].items():
                task_id = int(task_id_str)
                self.tasks[task_id] = Task.from_dict(task_dict)
        
        if 'next_task_id' in data:
            self.next_task_id = data['next_task_id']

class TaskTableModel(QStandardItemModel):
    """Table model for tasks."""
    
    def __init__(self, parent=None):
        """Initialize the model."""
        super().__init__(0, 6, parent)  # 6 columns
        self.setHorizontalHeaderLabels(["Title", "Date & Time", "Type", "Priority", "Status", "Assigned To"])
        self.tasks = []
        
    def set_tasks(self, tasks: List[Task]):
        """Set the task list and refresh the model."""
        self.tasks = sorted(tasks, key=lambda t: t.date_time)
        self.refresh()
        
    def refresh(self):
        """Refresh the model data."""
        self.removeRows(0, self.rowCount())
        
        for task in self.tasks:
            row = []
            
            # Title
            title_item = QStandardItem(task.title)
            title_item.setData(task.task_id, Qt.UserRole)  # Store task ID
            
            # Set background color based on status and whether overdue
            if task.status == TaskStatus.COMPLETED:
                title_item.setBackground(QBrush(QColor(240, 255, 240)))  # Light green
            elif task.status == TaskStatus.CANCELLED:
                title_item.setBackground(QBrush(QColor(240, 240, 240)))  # Light gray
            elif task.is_overdue():
                title_item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
                title_item.setToolTip("Overdue")
                
                # Make overdue tasks bold
                font = title_item.font()
                font.setBold(True)
                title_item.setFont(font)
            else:
                # Set background based on priority
                title_item.setBackground(QBrush(TaskPriority.to_color(task.priority)))
            
            row.append(title_item)
            
            # Date & Time
            date_time_str = task.date_time.strftime("%Y-%m-%d %I:%M %p")
            date_time_item = QStandardItem(date_time_str)
            date_time_item.setTextAlignment(Qt.AlignCenter)
            row.append(date_time_item)
            
            # Type
            type_item = QStandardItem(task.task_type)
            row.append(type_item)
            
            # Priority
            priority_item = QStandardItem(TaskPriority.to_string(task.priority))
            priority_item.setTextAlignment(Qt.AlignCenter)
            priority_item.setBackground(QBrush(TaskPriority.to_color(task.priority)))
            row.append(priority_item)
            
            # Status
            status_item = QStandardItem(TaskStatus.to_string(task.status))
            status_item.setTextAlignment(Qt.AlignCenter)
            row.append(status_item)
            
            # Assigned To
            assigned_to_item = QStandardItem(task.assigned_to or "")
            row.append(assigned_to_item)
            
            self.appendRow(row)

class SchedulerInterface(QWidget):
    """
    Scheduler Tab - provides a calendar and list view for scheduling tasks and activities.
    """
    
    # Signal emitted when a task is created or updated for a customer
    customer_task_updated = pyqtSignal(int, Task)  # customer_id, task

    def __init__(self, scheduler: TaskScheduler, notification_manager: NotificationManager, 
                 customer_repository: CustomerRepository):
        """
        Initializes the SchedulerInterface.

        Args:
            scheduler: TaskScheduler instance.
            notification_manager: NotificationManager instance for notifications.
            customer_repository: CustomerRepository for accessing customer data.
        """
        super().__init__()
        self.scheduler = scheduler
        self.notification_manager = notification_manager
        self.customer_repository = customer_repository
        
        # Cache for customer data to avoid excessive database queries
        self.customer_cache: Dict[int, Customer] = {}
        
        # Currently displayed date
        self.current_date = QDate.currentDate()
        
        # View mode: 'day', 'week', 'month'
        self.view_mode = 'day'
        
        # Async task management
        self.tasks = []
        
        self.setup_ui()
        
        # Initial data load
        self.load_schedule_data()
        self.load_customers_async()

    def setup_ui(self):
        """Sets up the user interface for the scheduler tab."""
        main_layout = QVBoxLayout(self)

        # Controls Bar (Buttons and View Options)
        controls_layout = QHBoxLayout()
        
        # Task Action buttons
        self.add_task_button = UIFactory.create_button("Add Task", self.add_new_task)
        self.add_task_button.setIcon(QIcon("icons/add_task.png"))
        self.edit_task_button = UIFactory.create_button("Edit Task", self.edit_selected_task)
        self.edit_task_button.setIcon(QIcon("icons/edit_task.png"))
        self.edit_task_button.setEnabled(False)  # Disabled until task is selected
        self.delete_task_button = UIFactory.create_button("Delete Task", self.delete_selected_task)
        self.delete_task_button.setIcon(QIcon("icons/delete_task.png"))
        self.delete_task_button.setEnabled(False)  # Disabled until task is selected
        
        controls_layout.addWidget(self.add_task_button)
        controls_layout.addWidget(self.edit_task_button)
        controls_layout.addWidget(self.delete_task_button)
        
        # Separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #888;")
        controls_layout.addWidget(separator)
        
        # View mode buttons
        self.day_view_button = UIFactory.create_button("Day", lambda: self.set_view_mode('day'))
        self.day_view_button.setIcon(QIcon("icons/day_view.png"))
        self.day_view_button.setCheckable(True)
        self.day_view_button.setChecked(self.view_mode == 'day')
        
        self.week_view_button = UIFactory.create_button("Week", lambda: self.set_view_mode('week'))
        self.week_view_button.setIcon(QIcon("icons/week_view.png"))
        self.week_view_button.setCheckable(True)
        self.week_view_button.setChecked(self.view_mode == 'week')
        
        self.month_view_button = UIFactory.create_button("Month", lambda: self.set_view_mode('month'))
        self.month_view_button.setIcon(QIcon("icons/month_view.png"))
        self.month_view_button.setCheckable(True)
        self.month_view_button.setChecked(self.view_mode == 'month')
        
        controls_layout.addWidget(self.day_view_button)
        controls_layout.addWidget(self.week_view_button)
        controls_layout.addWidget(self.month_view_button)
        
        # Filter/Search
        controls_layout.addStretch()
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Tasks", "Upcoming", "Overdue", "Completed", "Not Completed"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search tasks...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.filter_tasks)
        
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.filter_combo)
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_edit)
        
        main_layout.addLayout(controls_layout)
        
        # Main content - split between calendar and task list
        content_layout = QHBoxLayout()
        
        # Calendar view
        calendar_layout = QVBoxLayout()
        
        self.calendar_widget = QCalendarWidget()
        self.calendar_widget.setGridVisible(True)
        self.calendar_widget.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        self.calendar_widget.setHorizontalHeaderFormat(QCalendarWidget.SingleLetterDayNames)
        self.calendar_widget.setSelectedDate(self.current_date)
        self.calendar_widget.selectionChanged.connect(self.on_date_selected)
        
        # Customize calendar to highlight dates with tasks
        self.calendar_widget.setDateTextFormat(QDate.currentDate(), QTextCharFormat())  # Reset current date format
        
        calendar_layout.addWidget(self.calendar_widget)
        
        # Navigation controls below calendar
        calendar_nav_layout = QHBoxLayout()
        
        self.prev_button = UIFactory.create_button("◀ Prev", self.go_to_previous)
        self.prev_button.setFixedWidth(80)
        
        self.today_button = UIFactory.create_button("Today", self.go_to_today)
        self.today_button.setFixedWidth(80)
        
        self.next_button = UIFactory.create_button("Next ▶", self.go_to_next)
        self.next_button.setFixedWidth(80)
        
        calendar_nav_layout.addWidget(self.prev_button)
        calendar_nav_layout.addWidget(self.today_button)
        calendar_nav_layout.addWidget(self.next_button)
        
        calendar_layout.addLayout(calendar_nav_layout)
        
        content_layout.addLayout(calendar_layout, 1)  # 1:2 ratio for calendar:task list
        
        # Task list view
        task_list_layout = QVBoxLayout()
        
        # Date/period label
        self.date_label = QLabel()
        self.date_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.date_label.setAlignment(Qt.AlignCenter)
        self.update_date_label()
        
        task_list_layout.addWidget(self.date_label)
        
        # Task list
        self.task_table = QTableView()
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.task_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_table.setAlternatingRowColors(True)
        
        # Double-click to edit
        self.task_table.doubleClicked.connect(self.edit_selected_task)
        
        # Right-click context menu
        self.task_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.task_table.customContextMenuRequested.connect(self.show_context_menu)
        
        # Setup task model
        self.task_model = TaskTableModel()
        self.task_filter_model = QSortFilterProxyModel()
        self.task_filter_model.setSourceModel(self.task_model)
        self.task_filter_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.task_table.setModel(self.task_filter_model)
        
        # Selection changes
        self.task_table.selectionModel().selectionChanged.connect(self.on_task_selection_changed)
        
        task_list_layout.addWidget(self.task_table, 1)
        
        content_layout.addLayout(task_list_layout, 2)  # 1:2 ratio for calendar:task list
        
        main_layout.addLayout(content_layout, 1)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Task count label
        self.count_label = QLabel("0 tasks")
        status_layout.addWidget(self.count_label)
        
        main_layout.addLayout(status_layout)

    def set_view_mode(self, mode: str):
        """
        Set the view mode (day, week, month).
        
        Args:
            mode: The view mode to set
        """
        if mode not in ['day', 'week', 'month']:
            return
            
        self.view_mode = mode
        
        # Update button states
        self.day_view_button.setChecked(mode == 'day')
        self.week_view_button.setChecked(mode == 'week')
        self.month_view_button.setChecked(mode == 'month')
        
        # Update the UI to reflect the new view mode
        self.update_date_label()
        self.update_task_list()

    def update_date_label(self):
        """Update the date label based on view mode and current date."""
        if self.view_mode == 'day':
            self.date_label.setText(self.current_date.toString("dddd, MMMM d, yyyy"))
        elif self.view_mode == 'week':
            # Calculate week start and end dates
            day_of_week = self.current_date.dayOfWeek()
            week_start = self.current_date.addDays(-(day_of_week - 1))
            week_end = week_start.addDays(6)
            
            week_text = f"Week of {week_start.toString('MMM d')} - {week_end.toString('MMM d, yyyy')}"
            self.date_label.setText(week_text)
        elif self.view_mode == 'month':
            self.date_label.setText(self.current_date.toString("MMMM yyyy"))

    def update_task_list(self):
        """Update the task list based on the current view mode and date."""
        # Get tasks based on view mode
        if self.view_mode == 'day':
            # Convert QDate to Python date
            py_date = self.current_date.toPyDate()
            tasks = self.scheduler.get_tasks_for_date(py_date)
        elif self.view_mode == 'week':
            # Calculate week start and end dates
            day_of_week = self.current_date.dayOfWeek()
            week_start = self.current_date.addDays(-(day_of_week - 1))
            
            py_dates = [week_start.addDays(i).toPyDate() for i in range(7)]
            
            tasks = []
            for date in py_dates:
                tasks.extend(self.scheduler.get_tasks_for_date(date))
        elif self.view_mode == 'month':
            # Get first and last day of month
            year = self.current_date.year()
            month = self.current_date.month()
            first_day = QDate(year, month, 1)
            last_day = QDate(year, month, first_day.daysInMonth())
            
            # Convert to Python dates
            py_start_date = first_day.toPyDate()
            py_end_date = last_day.toPyDate()
            
            # Get all tasks and filter by date
            all_tasks = self.scheduler.get_all_tasks()
            tasks = [task for task in all_tasks if
                    py_start_date <= task.date_time.date() <= py_end_date]
        else:
            tasks = []
        
        # Apply current filter
        filter_index = self.filter_combo.currentIndex()
        if filter_index == 1:  # Upcoming
            now = datetime.now()
            tasks = [task for task in tasks if now <= task.date_time]
        elif filter_index == 2:  # Overdue
            now = datetime.now()
            tasks = [task for task in tasks if task.is_overdue()]
        elif filter_index == 3:  # Completed
            tasks = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        elif filter_index == 4:  # Not Completed
            tasks = [task for task in tasks if task.status != TaskStatus.COMPLETED]
        
        # Update the task model
        self.task_model.set_tasks(tasks)
        
        # Update count label
        self.count_label.setText(f"{len(tasks)} task{'' if len(tasks) == 1 else 's'}")
        
        # Reset the search filter
        self.search_edit.clear()

    def update_calendar_task_indicators(self):
        """Update the calendar with indicators for dates with tasks."""
        # Reset date formats
        self.calendar_widget.setDateTextFormat(QDate(), QTextCharFormat())
        
        # Get all tasks
        all_tasks = self.scheduler.get_all_tasks()
        
        # Group tasks by date
        task_dates = {}
        for task in all_tasks:
            date = task.date_time.date()
            if date not in task_dates:
                task_dates[date] = []
            task_dates[date].append(task)
        
        # Highlight dates with tasks
        for date, tasks in task_dates.items():
            qdate = QDate(date.year, date.month, date.day)
            
            # Skip dates outside the current month view
            if qdate.month() != self.calendar_widget.monthShown() or qdate.year() != self.calendar_widget.yearShown():
                continue
            
            # Create a format with background color
            text_format = QTextCharFormat()
            
            # Set background based on task status
            has_non_completed = any(task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] for task in tasks)
            has_overdue = any(task.is_overdue() for task in tasks)
            
            if has_overdue:
                text_format.setBackground(QBrush(QColor(255, 200, 200)))  # Light red for overdue
            elif has_non_completed:
                text_format.setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow for active tasks
            else:
                text_format.setBackground(QBrush(QColor(240, 255, 240)))  # Light green for completed tasks
            
            # Apply the format
            self.calendar_widget.setDateTextFormat(qdate, text_format)

    def on_date_selected(self):
        """Handle date selection in the calendar."""
        self.current_date = self.calendar_widget.selectedDate()
        self.update_date_label()
        self.update_task_list()

    def go_to_previous(self):
        """Go to previous day, week, or month based on view mode."""
        if self.view_mode == 'day':
            self.current_date = self.current_date.addDays(-1)
        elif self.view_mode == 'week':
            self.current_date = self.current_date.addDays(-7)
        elif self.view_mode == 'month':
            self.current_date = self.current_date.addMonths(-1)
        
        self.calendar_widget.setSelectedDate(self.current_date)
        self.update_date_label()
        self.update_task_list()

    def go_to_next(self):
        """Go to next day, week, or month based on view mode."""
        if self.view_mode == 'day':
            self.current_date = self.current_date.addDays(1)
        elif self.view_mode == 'week':
            self.current_date = self.current_date.addDays(7)
        elif self.view_mode == 'month':
            self.current_date = self.current_date.addMonths(1)
        
        self.calendar_widget.setSelectedDate(self.current_date)
        self.update_date_label()
        self.update_task_list()

    def go_to_today(self):
        """Go to today's date."""
        self.current_date = QDate.currentDate()
        self.calendar_widget.setSelectedDate(self.current_date)
        self.update_date_label()
        self.update_task_list()

    def on_task_selection_changed(self, selected, deselected):
        """Handle task selection changes in the list."""
        indexes = selected.indexes()
        
        if indexes:
            # Get the selected task ID
            proxy_index = indexes[0]
            source_index = self.task_filter_model.mapToSource(proxy_index)
            task_id = self.task_model.data(source_index.sibling(source_index.row(), 0), Qt.UserRole)
            
            if task_id is not None:
                self.edit_task_button.setEnabled(True)
                self.delete_task_button.setEnabled(True)
        else:
            self.edit_task_button.setEnabled(False)
            self.delete_task_button.setEnabled(False)

    def get_selected_task_id(self) -> Optional[int]:
        """Get the ID of the selected task."""
        # Get selected row
        indexes = self.task_table.selectionModel().selectedIndexes()
        if not indexes:
            return None
            
        # Get the task ID from the first column
        proxy_index = indexes[0]
        source_index = self.task_filter_model.mapToSource(proxy_index)
        task_id = self.task_model.data(source_index.sibling(source_index.row(), 0), Qt.UserRole)
        
        return task_id

    def add_new_task(self):
        """Open dialog to add a new task."""
        # Create a new task with the current date
        new_task = Task(
            title="",
            date_time=datetime.combine(
                self.current_date.toPyDate(),
                datetime.now().time()
            )
        )
        
        # Open dialog
        dialog = TaskDialog(self, new_task, list(self.customer_cache.values()))
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get task data from dialog
                task = dialog.get_task_data()
                
                # Add to scheduler
                task_id = self.scheduler.add_task(task)
                
                # Refresh UI
                self.update_task_list()
                self.update_calendar_task_indicators()
                
                # Show confirmation
                self.notification_manager.notify(
                    "Task Created",
                    f"Created task: {task.title}",
                    level="info"
                )
                
                # If task is associated with a customer, emit signal
                if task.customer_id is not None:
                    self.customer_task_updated.emit(task.customer_id, task)
                
            except Exception as e:
                ErrorHandler.handle_error(e, {"context": "SchedulerInterface.add_new_task"})

    def edit_selected_task(self):
        """Edit the selected task."""
        task_id = self.get_selected_task_id()
        if task_id is None:
            self.notification_manager.notify(
                "Edit Task",
                "No task selected",
                level="warning"
            )
            return
            
        # Get the task
        task = self.scheduler.get_task(task_id)
        if not task:
            self.notification_manager.notify(
                "Edit Task",
                f"Task with ID {task_id} not found",
                level="error"
            )
            return
            
        # Open dialog
        dialog = TaskDialog(self, task, list(self.customer_cache.values()))
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get updated task data from dialog
                updated_task = dialog.get_task_data()
                
                # Previous customer ID for signal emission
                previous_customer_id = task.customer_id
                
                # Update the task
                success = self.scheduler.update_task(updated_task)
                
                if success:
                    # Refresh UI
                    self.update_task_list()
                    self.update_calendar_task_indicators()
                    
                    # Show confirmation
                    self.notification_manager.notify(
                        "Task Updated",
                        f"Updated task: {updated_task.title}",
                        level="info"
                    )
                    
                    # If task customer changed or is associated with a customer, emit signal
                    if previous_customer_id is not None or updated_task.customer_id is not None:
                        # Emit signal for both old and new customer if changed
                        if previous_customer_id is not None and previous_customer_id != updated_task.customer_id:
                            self.customer_task_updated.emit(previous_customer_id, updated_task)
                        
                        if updated_task.customer_id is not None:
                            self.customer_task_updated.emit(updated_task.customer_id, updated_task)
                else:
                    self.notification_manager.notify(
                        "Task Update Failed",
                        f"Failed to update task with ID {task_id}",
                        level="error"
                    )
                
            except Exception as e:
                ErrorHandler.handle_error(e, {"context": "SchedulerInterface.edit_selected_task"})

    def delete_selected_task(self):
        """Delete the selected task."""
        task_id = self.get_selected_task_id()
        if task_id is None:
            self.notification_manager.notify(
                "Delete Task",
                "No task selected",
                level="warning"
            )
            return
            
        # Get the task
        task = self.scheduler.get_task(task_id)
        if not task:
            self.notification_manager.notify(
                "Delete Task",
                f"Task with ID {task_id} not found",
                level="error"
            )
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete task '{task.title}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Store customer ID for signal emission
                customer_id = task.customer_id
                
                # Delete the task
                success = self.scheduler.delete_task(task_id)
                
                if success:
                    # Refresh UI
                    self.update_task_list()
                    self.update_calendar_task_indicators()
                    
                    # Show confirmation
                    self.notification_manager.notify(
                        "Task Deleted",
                        f"Deleted task: {task.title}",
                        level="info"
                    )
                    
                    # If task was associated with a customer, emit signal
                    if customer_id is not None:
                        self.customer_task_updated.emit(customer_id, None)  # None to indicate deletion
                else:
                    self.notification_manager.notify(
                        "Task Deletion Failed",
                        f"Failed to delete task with ID {task_id}",
                        level="error"
                    )
                
            except Exception as e:
                ErrorHandler.handle_error(e, {"context": "SchedulerInterface.delete_selected_task"})

    def apply_filter(self):
        """Apply the selected filter to the task list."""
        self.update_task_list()

    def filter_tasks(self, text: str):
        """Filter tasks based on search text."""
        self.task_filter_model.setFilterFixedString(text)

    def show_context_menu(self, position):
        """
        Show context menu for tasks.
        
        Args:
            position: Position where the context menu should be shown
        """
        # Get the task ID
        task_id = self.get_selected_task_id()
        
        # Create the context menu
        menu = QMenu(self)
        
        if task_id is not None:
            # Get the task
            task = self.scheduler.get_task(task_id)
            
            if task:
                # Task actions
                edit_action = QAction("Edit Task", menu)
                edit_action.triggered.connect(self.edit_selected_task)
                menu.addAction(edit_action)
                
                # Status change actions
                menu.addSeparator()
                menu.addAction("Change Status")
                
                # Status submenu
                status_menu = QMenu("Change Status", menu)
                
                planned_action = QAction("Planned", status_menu)
                planned_action.triggered.connect(lambda: self.change_task_status(task_id, TaskStatus.PLANNED))
                planned_action.setEnabled(task.status != TaskStatus.PLANNED)
                status_menu.addAction(planned_action)
                
                in_progress_action = QAction("In Progress", status_menu)
                in_progress_action.triggered.connect(lambda: self.change_task_status(task_id, TaskStatus.IN_PROGRESS))
                in_progress_action.setEnabled(task.status != TaskStatus.IN_PROGRESS)
                status_menu.addAction(in_progress_action)
                
                completed_action = QAction("Completed", status_menu)
                completed_action.triggered.connect(lambda: self.change_task_status(task_id, TaskStatus.COMPLETED))
                completed_action.setEnabled(task.status != TaskStatus.COMPLETED)
                status_menu.addAction(completed_action)
                
                cancelled_action = QAction("Cancelled", status_menu)
                cancelled_action.triggered.connect(lambda: self.change_task_status(task_id, TaskStatus.CANCELLED))
                cancelled_action.setEnabled(task.status != TaskStatus.CANCELLED)
                status_menu.addAction(cancelled_action)
                
                menu.addMenu(status_menu)
                
                # Delete action
                menu.addSeparator()
                delete_action = QAction("Delete Task", menu)
                delete_action.triggered.connect(self.delete_selected_task)
                menu.addAction(delete_action)
            
        else:
            # No task selected
            add_action = QAction("Add New Task", menu)
            add_action.triggered.connect(self.add_new_task)
            menu.addAction(add_action)
        
        # View mode actions
        menu.addSeparator()
        
        day_action = QAction("Day View", menu)
        day_action.triggered.connect(lambda: self.set_view_mode('day'))
        day_action.setCheckable(True)
        day_action.setChecked(self.view_mode == 'day')
        menu.addAction(day_action)
        
        week_action = QAction("Week View", menu)
        week_action.triggered.connect(lambda: self.set_view_mode('week'))
        week_action.setCheckable(True)
        week_action.setChecked(self.view_mode == 'week')
        menu.addAction(week_action)
        
        month_action = QAction("Month View", menu)
        month_action.triggered.connect(lambda: self.set_view_mode('month'))
        month_action.setCheckable(True)
        month_action.setChecked(self.view_mode == 'month')
        menu.addAction(month_action)
        
        # Today action
        menu.addSeparator()
        today_action = QAction("Go to Today", menu)
        today_action.triggered.connect(self.go_to_today)
        menu.addAction(today_action)
        
        # Show the menu
        menu.exec_(self.task_table.viewport().mapToGlobal(position))

    def change_task_status(self, task_id: int, status: int):
        """
        Change the status of a task.
        
        Args:
            task_id: ID of the task to change
            status: New status
        """
        task = self.scheduler.get_task(task_id)
        if not task:
            return
            
        # Update status
        task.status = status
        task.updated_at = datetime.now()
        self.scheduler.update_task(task)
        
        # Refresh UI
        self.update_task_list()
        self.update_calendar_task_indicators()
        
        # Show confirmation
        status_str = TaskStatus.to_string(status)
        self.notification_manager.notify(
            "Task Status Changed",
            f"Changed status of '{task.title}' to {status_str}",
            level="info"
        )
        
        # If task is associated with a customer, emit signal
        if task.customer_id is not None:
            self.customer_task_updated.emit(task.customer_id, task)

    async def load_customers_async(self):
        """Asynchronously load customers for task assignment."""
        try:
            # Get customers from repository
            customers = await self.customer_repository.list_customers(
                limit=1000,  # Reasonable limit for now
                offset=0
            )
            
            # Update the customer cache
            self.customer_cache = {c.customer_id: c for c in customers}
            
            return customers
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "SchedulerInterface.load_customers_async"})
            return []

    def load_schedule_data(self):
        """Load schedule data and update the UI."""
        try:
            # Update task list and calendar
            self.update_task_list()
            self.update_calendar_task_indicators()
            
            # Load customers
            task = asyncio.create_task(self.load_customers_async())
            self.tasks.append(task)
            
            # Use task.add_done_callback to handle completion
            task.add_done_callback(lambda t: self.tasks.remove(t) if t in self.tasks else None)
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "SchedulerInterface.load_schedule_data"})

    def refresh(self):
        """Refresh the schedule view."""
        self.load_schedule_data()

    def add_task_for_customer(self, customer_id: int, task_title: str, task_date: datetime,
                            task_type: str = "Follow-up") -> Optional[Task]:
        """
        Add a task for a specific customer.
        
        Args:
            customer_id: ID of the customer
            task_title: Title for the task
            task_date: Date and time for the task
            task_type: Type of task
            
        Returns:
            The created task or None if creation failed
        """
        try:
            # Create the task
            task = Task(
                title=task_title,
                date_time=task_date,
                description=f"Customer follow-up task created on {datetime.now().strftime('%Y-%m-%d')}",
                task_type=task_type,
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PLANNED,
                customer_id=customer_id,
                assigned_to=None
            )
            
            # Add to scheduler
            task_id = self.scheduler.add_task(task)
            task.task_id = task_id
            
            # Refresh UI
            self.update_task_list()
            self.update_calendar_task_indicators()
            
            # Show confirmation
            self.notification_manager.notify(
                "Task Created for Customer",
                f"Created task: {task.title}",
                level="info"
            )
            
            return task
            
        except Exception as e:
            ErrorHandler.handle_error(e, {"context": "SchedulerInterface.add_task_for_customer"})
            return None

    def get_tasks_for_customer(self, customer_id: int) -> List[Task]:
        """
        Get all tasks for a specific customer.
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            List of tasks for the customer
        """
        return self.scheduler.get_tasks_for_customer(customer_id)
```

## app.py
```python
"""
Main entry point for the Electrician Estimator application.
"""

import sys
import os
import logging
import asyncio
import json
from pathlib import Path
from PyQt5 import QtWidgets
import qasync

from ui.main_window import MainWindow
from config.app_config import AppConfig
from services.ai_service import AIService
from services.computer_vision import ComputerVisionService
from services.measurement_service import MeasurementService
from services.estimate_generator import EstimateGeneratorService
from data.database_manager import DatabaseManager
from data.repositories.estimate_repository import EstimateRepository
from data.repositories.customer_repository import CustomerRepository
from data.repositories.image_repository import ImageRepository
from utils.notification_manager import NotificationManager
from utils.logging_utils import setup_logger
from utils.dependency_injection import ServiceContainer, ServiceLocator
from utils.error_handling import ErrorHandler
from utils.scheduler import Scheduler

# Initialize logging immediately
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
logger = setup_logger("app", log_dir=logs_dir, console_output=True)

def setup_exception_handling():
    """Set up global exception handling."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Don't catch KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception

def setup_environment():
    """Set up application environment and directories."""
    # Ensure all required directories exist
    directories = [
        Path("logs"),
        Path("data/image_storage"),
        Path("config"),
        Path("exports"),
        Path("cache")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Set current working directory to app root if needed
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        app_root = Path(sys.executable).parent
        os.chdir(app_root)

def save_app_state(container):
    """Save application state before exit."""
    try:
        # Get scheduler service for saving tasks
        scheduler = container.get(Scheduler)
        task_data = scheduler.save_tasks_to_dict()
        
        # Save tasks to file
        with open('data/tasks.json', 'w') as f:
            json.dump(task_data, f, indent=2, default=str)
        
        logger.info("Application state saved successfully")
    except Exception as e:
        logger.error(f"Error saving application state: {e}")

def load_app_state(container):
    """Load application state on startup."""
    try:
        # Get scheduler service
        scheduler = container.get(Scheduler)
        
        # Load tasks from file if exists
        task_file = Path('data/tasks.json')
        if task_file.exists():
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                scheduler.load_tasks_from_dict(task_data)
                logger.info(f"Loaded {len(task_data.get('tasks', {}))} tasks from saved state")
    except Exception as e:
        logger.error(f"Error loading application state: {e}")

def setup_dependency_container():
    """Set up the dependency injection container."""
    container = ServiceContainer()
    
    # Register services with appropriate lifecycle
    
    # Configuration - Singleton
    app_config = AppConfig()
    container.register(AppConfig, instance=app_config)
    
    # Notification manager - Singleton
    notification_manager = NotificationManager()
    container.register(NotificationManager, instance=notification_manager)
    
    # Initialize error handler
    ErrorHandler.initialize(notification_manager)
    
    # Task scheduler - Singleton
    scheduler = Scheduler()
    container.register(Scheduler, instance=scheduler)
    
    # Database manager - Singleton
    db_manager = DatabaseManager(
        db_type=app_config.get("DATABASE_TYPE", "sqlite"),
        db_path=app_config.get("DATABASE_PATH", "data/app.db")
    )
    container.register(DatabaseManager, instance=db_manager)
    
    # Repositories - Singleton
    container.register(EstimateRepository, factory=lambda c: EstimateRepository(c.get(DatabaseManager)), singleton=True)
    container.register(CustomerRepository, factory=lambda c: CustomerRepository(c.get(DatabaseManager)), singleton=True)
    container.register(ImageRepository, factory=lambda c: ImageRepository(c.get(DatabaseManager)), singleton=True)
    
    # Services - Some singletons, some transient
    container.register(AIService, singleton=True)  # Singleton due to API keys and rate limiting
    container.register(ComputerVisionService, singleton=True)  # Singleton for model loading
    container.register(MeasurementService, singleton=True)  # Singleton for calibration state
    
    # EstimateGeneratorService - Created on demand (transient)
    container.register(EstimateGeneratorService, singleton=False)
    
    # Initialize the service locator
    ServiceLocator.initialize(container)
    
    return container

def main():
    """Main application entry point."""
    logger.info("Starting Electrician Estimator Application...")
    
    # Set up application environment
    setup_environment()
    setup_exception_handling()
    
    # Create the Qt application
    app = qasync.QApplication(sys.argv)
    app.setApplicationName("Electrician Estimator")
    app.setOrganizationName("AJ Long Electric")
    
    # Set up dependency container
    try:
        container = setup_dependency_container()
        logger.info("Dependency container initialized")
        
        # Load saved application state
        load_app_state(container)
        
        # Get required services from container
        notification_manager = container.get(NotificationManager)
        
        # Create and show main window
        main_window = MainWindow(container)
        main_window.show()
        
        logger.info("Application UI started and visible")
        
        # Set up Qt event loop with asyncio integration
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # Handle application exit
        app.aboutToQuit.connect(lambda: save_app_state(container))
        
        # Run the application
        with loop:
            exit_code = loop.run_forever()
        
        return exit_code
    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        # Show error message to user
        error_dialog = QtWidgets.QMessageBox()
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Application Error")
        error_dialog.setText("Fatal error during application startup")
        error_dialog.setInformativeText(str(e))
        error_dialog.setDetailedText(f"Please check the log file at {logs_dir/'app.log'} for details.")
        error_dialog.exec_()
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## utils/background_processor.py
```python
"""
Background processing service for CPU-intensive operations
"""

import os
import time
import queue
import threading
import concurrent.futures
from typing import Callable, Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import traceback
from PyQt5.QtCore import QObject, pyqtSignal
from utils.logger import logger

class TaskPriority(Enum):
    """Task priority levels for the background processor."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Status values for background tasks."""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4

class BackgroundTask:
    """Represents a task submitted to the background processor."""
    
    def __init__(self, task_id: str, function: Callable, args: Tuple = (), 
                kwargs: Dict[str, Any] = None, priority: TaskPriority = TaskPriority.NORMAL,
                progress_callback: Optional[Callable[[int, str], None]] = None):
        """
        Initialize a background task.
        
        Args:
            task_id: Unique identifier for this task
            function: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority level
            progress_callback: Optional callback for progress updates
        """
        self.task_id = task_id
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.progress_callback = progress_callback
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.submit_time = time.time()
        self.start_time = None
        self.end_time = None
        
    def __lt__(self, other):
        """Compare tasks for priority queue ordering."""
        if not isinstance(other, BackgroundTask):
            return NotImplemented
        # Higher priority tasks come first
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        # Otherwise, first-come, first-served
        return self.submit_time < other.submit_time

class BackgroundProcessor(QObject):
    """
    Service for processing CPU-intensive tasks in the background.
    
    Uses a thread pool to execute tasks asynchronously, with progress
    reporting and priority-based scheduling.
    """
    # Signals for task status updates
    task_started = pyqtSignal(str)  # task_id
    task_completed = pyqtSignal(str, object)  # task_id, result
    task_failed = pyqtSignal(str, str)  # task_id, error message
    task_progress = pyqtSignal(str, int, str)  # task_id, progress percentage, message
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = "BgProc"):
        """
        Initialize the background processor.
        
        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count)
            thread_name_prefix: Prefix for worker thread names
        """
        super().__init__()
        self._max_workers = max_workers or os.cpu_count() or 4
        self._thread_name_prefix = thread_name_prefix
        self._executor = None
        self._task_queue = queue.PriorityQueue()
        self._active_tasks = {}  # task_id -> (future, task)
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread = None
        self._shutdown_event = threading.Event()
        
        # Start the processor
        self.start()
        
    def start(self):
        """Start the background processor."""
        with self._lock:
            if self._running:
                return
                
            self._running = True
            self._shutdown_event.clear()
            
            # Create thread pool executor
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=self._thread_name_prefix
            )
            
            # Start scheduler thread
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name=f"{self._thread_name_prefix}-Scheduler",
                daemon=True
            )
            self._scheduler_thread.start()
            
            logger.info(f"Background processor started with {self._max_workers} workers")
    
    def stop(self, wait: bool = True):
        """
        Stop the background processor.
        
        Args:
            wait: If True, wait for all tasks to complete; otherwise cancel pending tasks
        """
        with self._lock:
            if not self._running:
                return
                
            self._running = False
            self._shutdown_event.set()
            
            # Wait for scheduler to stop
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=2.0)
            
            # Handle executor shutdown
            if self._executor:
                if wait:
                    self._executor.shutdown(wait=True)
                else:
                    # Cancel all pending tasks
                    for future, task in self._active_tasks.values():
                        future.cancel()
                    
                    # Shut down executor without waiting
                    self._executor.shutdown(wait=False)
                
                self._executor = None
            
            self._active_tasks.clear()
            
            # Clear the task queue
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                    self._task_queue.task_done()
                except queue.Empty:
                    break
            
            logger.info("Background processor stopped")
    
    def submit_task(self, task_id: str, function: Callable, *args, 
                 priority: TaskPriority = TaskPriority.NORMAL,
                 progress_callback: Optional[Callable[[int, str], None]] = None,
                 **kwargs) -> str:
        """
        Submit a task for background processing.
        
        Args:
            task_id: Unique identifier for this task
            function: The function to execute
            *args: Positional arguments for the function
            priority: Task priority level
            progress_callback: Optional callback for progress updates
            **kwargs: Keyword arguments for the function
            
        Returns:
            The task ID
        """
        with self._lock:
            if not self._running:
                raise RuntimeError("Background processor is not running")
            
            # Create task object
            task = BackgroundTask(
                task_id=task_id,
                function=function,
                args=args,
                kwargs=kwargs,
                priority=priority,
                progress_callback=progress_callback
            )
            
            # Add to queue
            self._task_queue.put(task)
            
            logger.debug(f"Task submitted: {task_id} (priority: {priority.name})")
            return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            The task status or None if task not found
        """
        with self._lock:
            # Check active tasks
            if task_id in self._active_tasks:
                future, task = self._active_tasks[task_id]
                if future.cancelled():
                    return TaskStatus.CANCELLED
                elif future.done():
                    if future.exception() is not None:
                        return TaskStatus.FAILED
                    else:
                        return TaskStatus.COMPLETED
                else:
                    return TaskStatus.RUNNING
            
            # Check pending tasks in queue
            for task in list(self._task_queue.queue):
                if task.task_id == task_id:
                    return task.status
            
            # Task not found
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: The task ID
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        with self._lock:
            # Check active tasks
            if task_id in self._active_tasks:
                future, task = self._active_tasks[task_id]
                result = future.cancel()
                if result:
                    task.status = TaskStatus.CANCELLED
                    logger.debug(f"Task cancelled: {task_id}")
                return result
            
            # Check pending tasks in queue
            pending_tasks = list(self._task_queue.queue)
            for i, task in enumerate(pending_tasks):
                if task.task_id == task_id:
                    # Can't remove from queue directly, so mark it as cancelled
                    task.status = TaskStatus.CANCELLED
                    logger.debug(f"Pending task marked as cancelled: {task_id}")
                    return True
            
            # Task not found
            return False
    
    def _scheduler_loop(self):
        """Main loop for the task scheduler thread."""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next task from queue with timeout
                try:
                    task = self._task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    self._task_queue.task_done()
                    continue
                
                # Submit task to executor
                wrapped_task = self._wrap_task(task)
                future = self._executor.submit(wrapped_task)
                
                # Store active task
                with self._lock:
                    self._active_tasks[task.task_id] = (future, task)
                
                # Add done callback
                future.add_done_callback(lambda f, tid=task.task_id: self._task_completed(tid, f))
                
                # Mark queue task as done
                self._task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Sleep briefly to avoid tight loop in case of repeated errors
                time.sleep(0.1)
    
    def _wrap_task(self, task: BackgroundTask) -> Callable:
        """
        Wrap the task function with progress reporting and error handling.
        
        Args:
            task: The task to wrap
            
        Returns:
            A wrapped function that handles progress reporting and error handling
        """
        def wrapped_function():
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            # Emit task started signal
            self.task_started.emit(task.task_id)
            
            try:
                # Create progress callback that emits signal
                def progress_handler(percent: int, message: str = ""):
                    self.task_progress.emit(task.task_id, percent, message)
                    if task.progress_callback:
                        task.progress_callback(percent, message)
                
                # Add progress_callback to kwargs if function accepts it
                kwargs = task.kwargs.copy()
                if 'progress_callback' in task.function.__code__.co_varnames:
                    kwargs['progress_callback'] = progress_handler
                
                # Execute the task function
                result = task.function(*task.args, **kwargs)
                
                # Mark task as completed
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.end_time = time.time()
                
                return result
                
            except Exception as e:
                # Mark task as failed
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = time.time()
                
                # Log the error
                logger.error(f"Task failed: {task.task_id}", exc_info=True)
                
                # Re-raise the exception for the future
                raise
        
        return wrapped_function
    
    def _task_completed(self, task_id: str, future):
        """
        Handle task completion.
        
        Args:
            task_id: The task ID
            future: The completed future
        """
        with self._lock:
            if task_id not in self._active_tasks:
                return
                
            _, task = self._active_tasks[task_id]
            
            try:
                # Check if the task was cancelled
                if future.cancelled():
                    # Emit task failed signal
                    self.task_failed.emit(task_id, "Task was cancelled")
                    
                # Check if the task failed
                elif future.exception() is not None:
                    # Get the exception
                    exception = future.exception()
                    error_message = str(exception)
                    
                    # Emit task failed signal
                    self.task_failed.emit(task_id, error_message)
                    
                # Task completed successfully
                else:
                    # Get the result
                    result = future.result()
                    
                    # Emit task completed signal
                    self.task_completed.emit(task_id, result)
            finally:
                # Remove the task from active tasks
                del self._active_tasks[task_id]
    
    def get_status_summary(self) -> Dict[str, int]:
        """
        Get a summary of task status counts.
        
        Returns:
            Dictionary with counts by status
        """
        with self._lock:
            pending_count = 0
            running_count = len(self._active_tasks)
            cancelled_count = 0
            
            # Count pending and cancelled tasks in queue
            for task in list(self._task_queue.queue):
                if task.status == TaskStatus.CANCELLED:
                    cancelled_count += 1
                else:
                    pending_count += 1
            
            return {
                "pending": pending_count,
                "running": running_count,
                "cancelled": cancelled_count,
                "total": pending_count + running_count + cancelled_count
            }
```

## utils/performance_monitor.py
```python
"""
Performance monitoring utilities for tracking and optimizing application performance.
"""

import time
import threading
import functools
import statistics
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import psutil
from PyQt5.QtCore import QObject, pyqtSignal
from utils.logger import logger

class PerformanceMetric(Enum):
    """Types of performance metrics that can be tracked."""
    EXECUTION_TIME = "execution_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DATABASE_QUERIES = "database_queries"
    API_CALLS = "api_calls"
    IMAGE_PROCESSING = "image_processing"
    UI_RENDERING = "ui_rendering"

class PerformanceCategory(Enum):
    """Categories of operations for grouping metrics."""
    DATABASE = "database"
    AI_SERVICE = "ai_service"
    COMPUTER_VISION = "computer_vision"
    MEASUREMENT = "measurement"
    UI = "ui"
    FILE_IO = "file_io"
    NETWORK = "network"
    OTHER = "other"

class PerformanceMonitor(QObject):
    """
    Central service for monitoring and reporting application performance.
    """
    # Signals for performance alerts and updates
    performance_alert = pyqtSignal(str, str, float)  # category, operation, value
    performance_update = pyqtSignal(dict)  # performance data
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the performance monitor."""
        super().__init__()
        if PerformanceMonitor._instance is not None:
            raise RuntimeError("Use PerformanceMonitor.get_instance() to get the singleton instance")
            
        PerformanceMonitor._instance = self
        
        # Performance data
        self._metrics: Dict[str, Dict[str, List[float]]] = {
            category.value: {} for category in PerformanceCategory
        }
        
        # Thresholds for alerting
        self._thresholds: Dict[str, Dict[str, float]] = {
            category.value: {} for category in PerformanceCategory
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Process monitor for system metrics
        self._process = psutil.Process()
        
        # Sampling interval and history length
        self.sampling_interval = 1.0  # seconds
        self.history_length = 100  # samples
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Start background monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start background monitoring of system metrics."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_system_metrics,
            daemon=True,
            name="PerformanceMonitorThread"
        )
        self._monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=2.0)
        self._monitoring_thread = None
        logger.info("Performance monitoring stopped")

    def _monitor_system_metrics(self):
        """Background thread for monitoring system metrics."""
        while not self._stop_monitoring.is_set():
            try:
                # Gather CPU usage
                cpu_percent = self._process.cpu_percent(interval=0.1)
                self.record_metric(PerformanceCategory.OTHER.value, 
                                PerformanceMetric.CPU_USAGE.value,
                                cpu_percent)
                
                # Gather memory usage
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                self.record_metric(PerformanceCategory.OTHER.value,
                                PerformanceMetric.MEMORY_USAGE.value,
                                memory_mb)
                
                # Emit performance update
                summary = self.get_metrics_summary()
                self.performance_update.emit(summary)
                
                # Sleep for sampling interval
                self._stop_monitoring.wait(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in system metrics monitoring: {e}")
                # Sleep briefly to avoid tight loop on repeated errors
                time.sleep(1.0)

    def record_metric(self, category: str, operation: str, value: float) -> None:
        """
        Record a performance metric.
        
        Args:
            category: Category of the operation (use PerformanceCategory values)
            operation: Name of the operation
            value: Metric value to record
        """
        with self._lock:
            # Initialize metric list if it doesn't exist
            if category not in self._metrics:
                self._metrics[category] = {}
                
            if operation not in self._metrics[category]:
                self._metrics[category][operation] = []
                
            # Add the value and trim to history length
            metric_list = self._metrics[category][operation]
            metric_list.append(value)
            if len(metric_list) > self.history_length:
                metric_list.pop(0)
                
            # Check threshold
            self._check_threshold(category, operation, value)

    def _check_threshold(self, category: str, operation: str, value: float) -> None:
        """
        Check if a metric value exceeds its threshold and emit an alert.
        
        Args:
            category: Category of the operation
            operation: Name of the operation
            value: Metric value to check
        """
        if (category in self._thresholds and 
            operation in self._thresholds[category] and 
            value > self._thresholds[category][operation]):
            
            # Emit alert
            self.performance_alert.emit(category, operation, value)
            
            # Log alert
            threshold = self._thresholds[category][operation]
            logger.warning(f"Performance threshold exceeded: {category}/{operation} = "
                          f"{value:.2f} (threshold: {threshold:.2f})")

    def set_threshold(self, category: str, operation: str, threshold: float) -> None:
        """
        Set a performance threshold for alerting.
        
        Args:
            category: Category of the operation
            operation: Name of the operation
            threshold: Threshold value
        """
        with self._lock:
            if category not in self._thresholds:
                self._thresholds[category] = {}
                
            self._thresholds[category][operation] = threshold

    def get_metric_stats(self, category: str, operation: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            category: Category of the operation
            operation: Name of the operation
            
        Returns:
            Dictionary with metric statistics (min, max, avg, median, etc.)
        """
        with self._lock:
            if (category not in self._metrics or 
                operation not in self._metrics[category] or
                not self._metrics[category][operation]):
                return {
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "median": 0,
                    "count": 0,
                    "last": 0
                }
                
            values = self._metrics[category][operation]
            
            return {
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "count": len(values),
                "last": values[-1]
            }

    def get_metrics_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with categories, operations, and their statistics
        """
        with self._lock:
            summary = {}
            
            for category, operations in self._metrics.items():
                summary[category] = {}
                
                for operation in operations:
                    summary[category][operation] = self.get_metric_stats(category, operation)
                    
            return summary

    def time_operation(self, category: str, operation: str):
        """
        Decorator for timing an operation's execution.
        
        Args:
            category: Category of the operation
            operation: Name of the operation
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    return func(*args, **kwargs)
                finally:
                    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    self.record_metric(category, operation, execution_time)
                    
            return wrapper
        return decorator

    async def time_async_operation(self, category: str, operation: str, coro):
        """
        Timing wrapper for an asynchronous coroutine.
        
        Args:
            category: Category of the operation
            operation: Name of the operation
            coro: Coroutine to execute and time
            
        Returns:
            Result of the coroutine
        """
        start_time = time.time()
        
        try:
            return await coro
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.record_metric(category, operation, execution_time)

    def clear_metrics(self, category: Optional[str] = None, operation: Optional[str] = None) -> None:
        """
        Clear metrics data.
        
        Args:
            category: Optional category to clear (all categories if None)
            operation: Optional operation to clear (all operations if None)
        """
        with self._lock:
            if category is None:
                # Clear all metrics
                self._metrics = {
                    cat.value: {} for cat in PerformanceCategory
                }
            elif operation is None:
                # Clear all operations in the category
                if category in self._metrics:
                    self._metrics[category] = {}
            else:
                # Clear specific operation
                if category in self._metrics and operation in self._metrics[category]:
                    self._metrics[category][operation] = []

    def generate_performance_report(self) -> str:
        """
        Generate a formatted performance report.
        
        Returns:
            Formatted string with performance metrics
        """
        summary = self.get_metrics_summary()
        
        # Build report
        report = ["Performance Report", "================="]
        
        for category, operations in summary.items():
            if not operations:
                continue
                
            report.append(f"\n{category.upper()}")
            report.append("-" * len(category))
            
            for operation, stats in operations.items():
                report.append(f"  {operation}:")
                report.append(f"    Last: {stats['last']:.2f}")
                report.append(f"    Avg: {stats['avg']:.2f}")
                report.append(f"    Min: {stats['min']:.2f}")
                report.append(f"    Max: {stats['max']:.2f}")
                report.append(f"    Count: {stats['count']}")
                
        return "\n".join(report)

# Create a singleton instance
performance_monitor = PerformanceMonitor.get_instance()

# Convenience decorator for timing non-async functions
def monitor_performance(category: Union[str, PerformanceCategory], 
                       operation: Union[str, PerformanceMetric]):
    """
    Decorator for monitoring the performance of a function.
    
    Args:
        category: Category for the operation (string or PerformanceCategory)
        operation: Name of the operation (string or PerformanceMetric)
        
    Returns:
        Decorated function with performance monitoring
    """
    # Convert enum values to strings if needed
    if isinstance(category, PerformanceCategory):
        category = category.value
    if isinstance(operation, PerformanceMetric):
        operation = operation.value
        
    return performance_monitor.time_operation(category, operation)

# Convenience function for async monitoring
async def monitor_async_performance(category: Union[str, PerformanceCategory],
                                  operation: Union[str, PerformanceMetric],
                                  coro):
    """
    Monitor the performance of an async coroutine.
    
    Args:
        category: Category for the operation (string or PerformanceCategory)
        operation: Name of the operation (string or PerformanceMetric)
        coro: Coroutine to monitor
        
    Returns:
        Result of the coroutine
    """
    # Convert enum values to strings if needed
    if isinstance(category, PerformanceCategory):
        category = category.value
    if isinstance(operation, PerformanceMetric):
        operation = operation.value
        
    return await performance_monitor.time_async_operation(category, operation, coro)
```

## pyinstaller_build.py
```python
"""
PyInstaller build script for creating executable distributions.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import platform

def parse_arguments():
    """Parse command line arguments for the build script."""
    parser = argparse.ArgumentParser(description='Build executable distribution with PyInstaller')
    
    parser.add_argument('--clean', action='store_true', help='Clean build directories before building')
    parser.add_argument('--onefile', action='store_true', help='Create a single executable file (slower startup)')
    parser.add_argument('--console', action='store_true', help='Show console window (for debugging)')
    parser.add_argument('--name', default="ElectricianEstimator", help='Application name')
    parser.add_argument('--version', default="1.0.0", help='Application version')
    parser.add_argument('--icon', default="icons/app_icon.ico", help='Application icon path')
    
    return parser.parse_args()

def clean_build_dirs():
    """Clean build directories."""
    print("Cleaning build directories...")
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Removing {dir_name}...")
            shutil.rmtree(dir_name)
    
    # Remove .spec files
    for spec_file in Path('.').glob('*.spec'):
        print(f"Removing {spec_file}...")
        spec_file.unlink()

def create_version_info(app_name, app_version):
    """Create version info file for Windows builds."""
    version_path = Path('version_info.txt')
    version_parts = app_version.split('.')
    while len(version_parts) < 4:
        version_parts.append('0')
    
    version_str = '.'.join(version_parts)
    file_version = ','.join(version_parts)
    
    with open(version_path, 'w') as f:
        f.write(f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({file_version}),
    prodvers=({file_version}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'AJ Long Electric'),
          StringStruct(u'FileDescription', u'{app_name}'),
          StringStruct(u'FileVersion', u'{version_str}'),
          StringStruct(u'InternalName', u'{app_name}'),
          StringStruct(u'LegalCopyright', u'Copyright © 2024 AJ Long Electric'),
          StringStruct(u'OriginalFilename', u'{app_name}.exe'),
          StringStruct(u'ProductName', u'{app_name}'),
          StringStruct(u'ProductVersion', u'{version_str}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
""")
    return version_path

def build_executable(args):
    """Build the executable with PyInstaller."""
    print(f"Building {args.name} version {args.version}...")
    
    # Prepare build command
    cmd = ['pyinstaller', '--clean', '--noconfirm']
    
    # Add icon if exists
    if os.path.exists(args.icon):
        cmd.extend(['--icon', args.icon])
    else:
        print(f"Warning: Icon file {args.icon} not found.")
    
    # Add windowed/console mode
    if not args.console:
        cmd.append('--windowed')
    
    # Add onefile/onedir mode
    if args.onefile:
        cmd.append('--onefile')
    else:
        cmd.append('--onedir')
    
    # Add version info for Windows
    if platform.system() == 'Windows':
        version_file = create_version_info(args.name, args.version)
        cmd.extend(['--version-file', str(version_file)])
    
    # Add name
    cmd.extend(['--name', args.name])
    
    # Add hidden imports
    cmd.extend(['--hidden-import', 'PyQt5.sip'])
    
    # Add data files
    cmd.extend(['--add-data', 'icons/*:icons'])
    
    # Add main script
    cmd.append('app.py')
    
    # Run PyInstaller
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("PyInstaller stdout:")
    print(process.stdout)
    
    if process.stderr:
        print("PyInstaller stderr:")
        print(process.stderr)
    
    # Check if build was successful
    if process.returncode != 0:
        print(f"PyInstaller build failed with code {process.returncode}")
        return False
    
    return True

def copy_additional_files(args):
    """Copy additional files to the distribution directory."""
    print("Copying additional files...")
    
    dist_dir = Path('dist')
    if args.onefile:
        target_dir = dist_dir
    else:
        target_dir = dist_dir / args.name
    
    # Create directories
    (target_dir / 'logs').mkdir(exist_ok=True)
    (target_dir / 'data').mkdir(exist_ok=True)
    (target_dir / 'exports').mkdir(exist_ok=True)
    
    # Copy README and license if they exist
    for file in ['README.md', 'LICENSE']:
        if os.path.exists(file):
            shutil.copy(file, target_dir)
    
    print(f"Additional files copied to {target_dir}")

def create_installer_script(args):
    """Create an NSIS installer script for Windows."""
    if platform.system() != 'Windows':
        return None
    
    print("Creating installer script...")
    
    installer_script = Path('installer.nsi')
    version_parts = args.version.split('.')
    while len(version_parts) < 3:
        version_parts.append('0')
    
    version_str = '.'.join(version_parts)
    
    with open(installer_script, 'w') as f:
        f.write(f"""
!define APPNAME "{args.name}"
!define COMPANYNAME "AJ Long Electric"
!define DESCRIPTION "AI-Powered Electrical Estimation App"
!define VERSIONMAJOR {version_parts[0]}
!define VERSIONMINOR {version_parts[1]}
!define VERSIONBUILD {version_parts[2]}
!define HELPURL "https://www.example.com/help"
!define UPDATEURL "https://www.example.com/update"
!define ABOUTURL "https://www.example.com/about"

!define INSTALLER_NAME "${{APPNAME}}-${{VERSIONMAJOR}}.${{VERSIONMINOR}}.${{VERSIONBUILD}}-setup.exe"

Name "${{APPNAME}}"
OutFile "dist\\${{INSTALLER_NAME}}"
InstallDir "$PROGRAMFILES\\${{COMPANYNAME}}\\${{APPNAME}}"
InstallDirRegKey HKLM "Software\\${{COMPANYNAME}}\\${{APPNAME}}" "Install_Dir"
RequestExecutionLevel admin

!include "MUI2.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON "{args.icon}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    
    # Files to install
    File /r "dist\\{args.name}\\*.*"
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\\uninstall.exe"
    
    # Start Menu
    CreateDirectory "$SMPROGRAMS\\${{COMPANYNAME}}"
    CreateShortCut "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\{args.name}.exe"
    CreateShortCut "$SMPROGRAMS\\${{COMPANYNAME}}\\Uninstall ${{APPNAME}}.lnk" "$INSTDIR\\uninstall.exe"
    
    # Registry keys for Add/Remove Programs
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayName" "${{COMPANYNAME}} ${{APPNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayIcon" "$INSTDIR\\{args.name}.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "Publisher" "${{COMPANYNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayVersion" "${{VERSIONMAJOR}}.${{VERSIONMINOR}}.${{VERSIONBUILD}}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMajor" ${{VERSIONMAJOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMinor" ${{VERSIONMINOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoRepair" 1
SectionEnd

Section "Uninstall"
    # Remove program files
    RMDir /r "$INSTDIR"
    
    # Remove Start Menu items
    Delete "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk"
    Delete "$SMPROGRAMS\\${{COMPANYNAME}}\\Uninstall ${{APPNAME}}.lnk"
    RMDir "$SMPROGRAMS\\${{COMPANYNAME}}"
    
    # Remove registry keys
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}"
    DeleteRegKey HKLM "Software\\${{COMPANYNAME}}\\${{APPNAME}}"
SectionEnd
""")
    
    return installer_script

def build_installer(installer_script):
    """Build the installer with NSIS."""
    if not installer_script or not installer_script.exists():
        return False
    
    print("Building installer...")
    
    makensis_path = shutil.which('makensis')
    if not makensis_path:
        print("NSIS not found. Skipping installer creation.")
        return False
    
    cmd = [makensis_path, str(installer_script)]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("NSIS stdout:")
    print(process.stdout)
    
    if process.stderr:
        print("NSIS stderr:")
        print(process.stderr)
    
    # Check if build was successful
    if process.returncode != 0:
        print(f"NSIS build failed with code {process.returncode}")
        return False
    
    return True

def main():
    """Main entry point for the build script."""
    args = parse_arguments()
    
    # Check if PyInstaller is installed
    if shutil.which('pyinstaller') is None:
        print("PyInstaller not found. Install with 'pip install pyinstaller'.")
        return 1
    
    # Record start time
    start_time = datetime.now()
    print(f"Build started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clean directories if requested
    if args.clean:
        clean_build_dirs()
    
    # Build executable
    if not build_executable(args):
        return 1
    
    # Copy additional files
    copy_additional_files(args)
    
    # Create and build installer (Windows only)
    if platform.system() == 'Windows' and not args.onefile:
        installer_script = create_installer_script(args)
        build_installer(installer_script)
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Build completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total build time: {duration}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

# Enhanced UI/UX and Wizard-like Workflows Implementation

I'll implement the wizard-like workflows as requested, focusing on creating a modular, extensible framework that can be used for various estimation tasks.

## First, let's create a base wizard framework:

### ui/wizards/wizard_base.py
```python
"""
Base classes and components for implementing wizard-like workflows in the application.
"""

from typing import List, Dict, Any, Optional, Callable, Type
from enum import Enum
from PyQt5.QtWidgets import (QWidget, QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QProgressBar, QSpacerItem, QSizePolicy,
                           QMessageBox, QCheckBox, QComboBox, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPainter

from utils.logger import logger
from utils.notification_manager import NotificationManager
from utils.error_handling import ErrorHandler

class WizardStep(Enum):
    """Enumeration of possible wizard step states."""
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    ERROR = 3
    SKIPPED = 4

class WizardStepData:
    """Data class for tracking wizard step state and data."""
    
    def __init__(self, step_id: str, title: str, description: str = "", 
                 optional: bool = False, help_text: str = ""):
        """
        Initialize wizard step data.
        
        Args:
            step_id: Unique identifier for this step
            title: Step title displayed in the wizard
            description: Longer description of the step
            optional: Whether this step can be skipped
            help_text: Help text to guide users
        """
        self.step_id = step_id
        self.title = title
        self.description = description
        self.optional = optional
        self.help_text = help_text
        self.state = WizardStep.NOT_STARTED
        self.data: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
        self.completion_percentage: int = 0
    
    def reset(self):
        """Reset the step state."""
        self.state = WizardStep.NOT_STARTED
        self.data = {}
        self.error_message = None
        self.completion_percentage = 0
    
    def is_completed(self) -> bool:
        """Check if the step is completed."""
        return self.state == WizardStep.COMPLETED
    
    def is_optional(self) -> bool:
        """Check if the step is optional."""
        return self.optional
    
    def has_error(self) -> bool:
        """Check if the step has an error."""
        return self.state == WizardStep.ERROR
    
    def is_skipped(self) -> bool:
        """Check if the step is skipped."""
        return self.state == WizardStep.SKIPPED

class CustomWizardPage(QWizardPage):
    """Custom wizard page with enhanced functionality."""
    
    # Signal emitted when page is completed
    page_completed = pyqtSignal(bool)  # Success flag
    
    # Signal for progress updates
    progress_updated = pyqtSignal(int, str)  # Percentage, message
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the wizard page.
        
        Args:
            wizard_data: Data for this wizard step
            parent: Parent widget
        """
        super().__init__(parent)
        self.wizard_data = wizard_data
        self.setup_ui()
        
        # Connect progress signal to update the progress bar
        self.progress_updated.connect(self.update_progress)
    
    def setup_ui(self):
        """Set up the wizard page UI."""
        # Set page title and subtitle
        self.setTitle(self.wizard_data.title)
        self.setSubTitle(self.wizard_data.description)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Help text (if provided)
        if self.wizard_data.help_text:
            help_label = QLabel(self.wizard_data.help_text)
            help_label.setWordWrap(True)
            help_label.setStyleSheet("color: #666666; font-style: italic;")
            layout.addWidget(help_label)
            layout.addSpacing(10)
        
        # Content area - to be overridden by subclasses
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        layout.addWidget(self.content_widget, 1)
        
        # Progress area
        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        self.progress_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(self.progress_widget)
        
        # Skip checkbox for optional steps
        if self.wizard_data.optional:
            self.skip_checkbox = QCheckBox("Skip this step")
            self.skip_checkbox.toggled.connect(self.on_skip_toggled)
            layout.addWidget(self.skip_checkbox)
        
        # Initially hide progress area until needed
        self.progress_widget.setVisible(False)
    
    def update_progress(self, percentage: int, message: str = ""):
        """
        Update the progress display.
        
        Args:
            percentage: Completion percentage (0-100)
            message: Status message
        """
        # Show progress widget if hidden
        if not self.progress_widget.isVisible():
            self.progress_widget.setVisible(True)
        
        # Update progress bar and label
        self.progress_bar.setValue(percentage)
        if message:
            self.progress_label.setText(message)
        
        # Store progress in wizard data
        self.wizard_data.completion_percentage = percentage
        
        # Process events to ensure UI updates
        QApplication.processEvents()
    
    def on_skip_toggled(self, checked: bool):
        """
        Handle skip checkbox toggle.
        
        Args:
            checked: Whether the checkbox is checked
        """
        # Update wizard data state
        if checked:
            self.wizard_data.state = WizardStep.SKIPPED
        else:
            # Reset to appropriate state
            if self.wizard_data.completion_percentage == 100:
                self.wizard_data.state = WizardStep.COMPLETED
            elif self.wizard_data.error_message:
                self.wizard_data.state = WizardStep.ERROR
            else:
                self.wizard_data.state = WizardStep.NOT_STARTED
        
        # Update UI based on skip state
        self.content_widget.setEnabled(not checked)
        
        # Emit completion signal if needed
        if checked:
            self.page_completed.emit(True)
        
        # Force completeChanged to re-evaluate next button state
        self.completeChanged.emit()
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete and the wizard can proceed.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if:
        # 1. Step is marked as completed or skipped, or
        # 2. Step is optional and skip checkbox is checked
        return (self.wizard_data.is_completed() or 
                self.wizard_data.is_skipped() or 
                (self.wizard_data.is_optional() and 
                 hasattr(self, 'skip_checkbox') and 
                 self.skip_checkbox.isChecked()))
    
    def show_error(self, message: str):
        """
        Display an error message on the page.
        
        Args:
            message: Error message to display
        """
        # Update wizard data
        self.wizard_data.state = WizardStep.ERROR
        self.wizard_data.error_message = message
        
        # Show error in UI
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.exec_()
        
        # Update progress area
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff6666; }")
        self.progress_label.setText("Error")
        
        # Force completeChanged to re-evaluate next button state
        self.completeChanged.emit()
    
    def mark_completed(self, success: bool = True):
        """
        Mark the page as completed.
        
        Args:
            success: Whether completion was successful
        """
        if success:
            self.wizard_data.state = WizardStep.COMPLETED
            self.update_progress(100, "Complete")
        else:
            self.wizard_data.state = WizardStep.ERROR
        
        # Emit completion signal
        self.page_completed.emit(success)
        
        # Force completeChanged to re-evaluate next button state
        self.completeChanged.emit()

class BaseWizard(QWizard):
    """Base wizard framework for guided workflows."""
    
    # Signal emitted when wizard completes
    wizard_completed = pyqtSignal(bool, dict)  # Success flag, result data
    
    def __init__(self, title: str, parent=None):
        """
        Initialize the wizard.
        
        Args:
            title: Wizard title
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.HaveHelpButton, True)
        
        # Set minimum size
        self.setMinimumSize(800, 600)
        
        # Wizard data dictionary
        self.wizard_data: Dict[str, WizardStepData] = {}
        
        # Steps dictionary (step_id -> page_id)
        self.steps: Dict[str, int] = {}
        
        # Connect signals
        self.helpRequested.connect(self.show_help)
        self.finished.connect(self.on_wizard_finished)
        
        # Style the wizard
        self.setStyleSheet("""
            QWizard {
                background-color: #f8f8f8;
            }
            QWizard QLabel {
                font-size: 12px;
            }
            QWizardPage {
                background-color: white;
            }
            QWizard QLabel.title {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
        """)
    
    def add_step(self, step_data: WizardStepData, page_class: Type[CustomWizardPage]) -> int:
        """
        Add a step to the wizard.
        
        Args:
            step_data: Data for the wizard step
            page_class: Wizard page class to instantiate
            
        Returns:
            Page ID for the added step
        """
        # Store step data
        self.wizard_data[step_data.step_id] = step_data
        
        # Create page instance
        page = page_class(step_data, self)
        
        # Add page to wizard
        page_id = self.addPage(page)
        
        # Connect page signals
        page.page_completed.connect(lambda success: self.on_step_completed(step_data.step_id, success))
        
        # Store step mapping
        self.steps[step_data.step_id] = page_id
        
        return page_id
    
    def get_page_for_step(self, step_id: str) -> Optional[CustomWizardPage]:
        """
        Get the page for a specific step.
        
        Args:
            step_id: Step ID
            
        Returns:
            Wizard page or None if not found
        """
        if step_id in self.steps:
            page_id = self.steps[step_id]
            return self.page(page_id)
        return None
    
    def on_step_completed(self, step_id: str, success: bool):
        """
        Handle step completion event.
        
        Args:
            step_id: Step ID
            success: Whether completion was successful
        """
        logger.info(f"Wizard step {step_id} completed with success={success}")
        
        # Update step state
        step_data = self.wizard_data.get(step_id)
        if step_data:
            step_data.state = WizardStep.COMPLETED if success else WizardStep.ERROR
        
        # Navigate to next step if configured
        if success and self.currentPage().isFinalPage() is False:
            self.next()
    
    def show_help(self):
        """Show help for the current step."""
        current_id = self.currentId()
        for step_id, page_id in self.steps.items():
            if page_id == current_id:
                step_data = self.wizard_data.get(step_id)
                if step_data and step_data.help_text:
                    QMessageBox.information(self, "Help", step_data.help_text)
                    return
        
        # Default help if no specific help found
        QMessageBox.information(self, "Help", "This wizard guides you through the estimation process step by step.")
    
    def on_wizard_finished(self, result: int):
        """
        Handle wizard completion.
        
        Args:
            result: Wizard result code
        """
        if result == QDialog.Accepted:
            # Build result data from all steps
            result_data = {}
            for step_id, step_data in self.wizard_data.items():
                if not step_data.is_skipped():
                    result_data[step_id] = step_data.data
            
            logger.info(f"Wizard completed successfully")
            self.wizard_completed.emit(True, result_data)
        else:
            logger.info(f"Wizard cancelled")
            self.wizard_completed.emit(False, {})
    
    def get_step_data(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the data for a specific step.
        
        Args:
            step_id: Step ID
            
        Returns:
            Step data or None if not found
        """
        step_data = self.wizard_data.get(step_id)
        if step_data:
            return step_data.data
        return None
    
    def set_step_data(self, step_id: str, data: Dict[str, Any]):
        """
        Set the data for a specific step.
        
        Args:
            step_id: Step ID
            data: Step data
        """
        step_data = self.wizard_data.get(step_id)
        if step_data:
            step_data.data = data
    
    def reset_wizard(self):
        """Reset the wizard to its initial state."""
        # Reset all step data
        for step_data in self.wizard_data.values():
            step_data.reset()
        
        # Reset to first page
        self.restart()

class WizardManager:
    """
    Manager class for registering and launching wizards.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the wizard manager."""
        if WizardManager._instance is not None:
            raise RuntimeError("Use WizardManager.get_instance() to get the singleton instance")
        
        # Dictionary of registered wizards
        self.registered_wizards: Dict[str, Type[BaseWizard]] = {}
        
        # Notification manager for user feedback
        self.notification_manager = NotificationManager()
    
    def register_wizard(self, wizard_id: str, wizard_class: Type[BaseWizard]):
        """
        Register a wizard class.
        
        Args:
            wizard_id: Unique identifier for the wizard
            wizard_class: Wizard class to register
        """
        self.registered_wizards[wizard_id] = wizard_class
        logger.info(f"Registered wizard: {wizard_id}")
    
    def launch_wizard(self, wizard_id: str, parent=None) -> Optional[Dict[str, Any]]:
        """
        Launch a registered wizard.
        
        Args:
            wizard_id: ID of the wizard to launch
            parent: Parent widget
            
        Returns:
            Wizard result data or None if cancelled
        """
        if wizard_id not in self.registered_wizards:
            logger.error(f"Wizard not found: {wizard_id}")
            self.notification_manager.notify(
                "Error", 
                f"Wizard '{wizard_id}' not found", 
                level="error"
            )
            return None
        
        try:
            # Create wizard instance
            wizard_class = self.registered_wizards[wizard_id]
            wizard = wizard_class(parent)
            
            # Execute wizard
            result = wizard.exec_()
            
            if result == QDialog.Accepted:
                # Build result data from all steps
                result_data = {}
                for step_id, step_data in wizard.wizard_data.items():
                    if not step_data.is_skipped():
                        result_data[step_id] = step_data.data
                
                return result_data
        except Exception as e:
            logger.error(f"Error launching wizard {wizard_id}: {e}", exc_info=True)
            ErrorHandler.handle_error(e, {"context": f"WizardManager.launch_wizard({wizard_id})"})
        
        return None

# Create singleton instance
wizard_manager = WizardManager.get_instance()
```

## Now, let's implement a specific room-by-room estimation wizard:

### ui/wizards/room_by_room_wizard.py
```python
"""
Room-by-room estimation wizard for guiding users through a structured estimation process.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QListWidget, QListWidgetItem, QLineEdit, QComboBox, 
                           QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
                           QFileDialog, QMessageBox, QScrollArea, QGridLayout, QFrame,
                           QTabWidget, QTextEdit, QToolButton)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont

from ui.wizards.wizard_base import BaseWizard, CustomWizardPage, WizardStepData
from models.customer import Customer
from models.estimate import Estimate, CustomerInfo, LineItem
from data.repositories.customer_repository import CustomerRepository
from data.repositories.estimate_repository import EstimateRepository
from services.estimate_generator import EstimateGeneratorService
from utils.logger import logger
from utils.error_handling import ErrorHandler
from utils.dependency_injection import ServiceLocator

class RoomType(Enum):
    """Types of rooms in a building."""
    KITCHEN = "Kitchen"
    BATHROOM = "Bathroom"
    BEDROOM = "Bedroom"
    LIVING_ROOM = "Living Room"
    DINING_ROOM = "Dining Room"
    FAMILY_ROOM = "Family Room"
    OFFICE = "Office"
    GARAGE = "Garage"
    BASEMENT = "Basement"
    UTILITY = "Utility Room"
    HALLWAY = "Hallway"
    OTHER = "Other"

class ElectricalComponent(Enum):
    """Types of electrical components for estimation."""
    RECEPTACLE = "Receptacle Outlet"
    SWITCH = "Switch"
    LIGHT_FIXTURE = "Light Fixture"
    CEILING_FAN = "Ceiling Fan"
    GFCI_OUTLET = "GFCI Outlet"
    AFCI_OUTLET = "AFCI Outlet"
    DIMMER_SWITCH = "Dimmer Switch"
    SMART_SWITCH = "Smart Switch"
    MOTION_SENSOR = "Motion Sensor"
    SMOKE_DETECTOR = "Smoke Detector"
    CARBON_MONOXIDE = "Carbon Monoxide Detector"
    THERMOSTAT = "Thermostat"
    DOORBELL = "Doorbell"
    APPLIANCE = "Appliance Connection"
    SUBPANEL = "Subpanel"
    CIRCUIT = "New Circuit"
    WIRE_RUN = "Wire Run"
    CONDUIT = "Conduit"
    OTHER = "Other Component"

class Room:
    """Data model for a room in the estimation."""
    
    def __init__(self, name: str, room_type: RoomType):
        """
        Initialize a room.
        
        Args:
            name: Room name
            room_type: Type of room
        """
        self.name = name
        self.room_type = room_type
        self.components: List[Dict[str, Any]] = []
        self.notes: str = ""
        self.measurements: Dict[str, float] = {
            "length": 0,
            "width": 0,
            "height": 0,
        }
        self.images: List[str] = []
    
    def add_component(self, component_type: ElectricalComponent, quantity: int, notes: str = "", 
                     unit_price: float = 0, labor_hours: float = 0):
        """
        Add an electrical component to the room.
        
        Args:
            component_type: Type of component
            quantity: Number of components
            notes: Additional notes
            unit_price: Price per unit
            labor_hours: Labor hours per unit
        """
        self.components.append({
            "type": component_type,
            "quantity": quantity,
            "notes": notes,
            "unit_price": unit_price,
            "labor_hours": labor_hours
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert room to dictionary.
        
        Returns:
            Dictionary representation of the room
        """
        return {
            "name": self.name,
            "room_type": self.room_type.value,
            "components": [
                {
                    "type": comp["type"].value,
                    "quantity": comp["quantity"],
                    "notes": comp["notes"],
                    "unit_price": comp["unit_price"],
                    "labor_hours": comp["labor_hours"]
                }
                for comp in self.components
            ],
            "notes": self.notes,
            "measurements": self.measurements,
            "images": self.images
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        """
        Create a room from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Room instance
        """
        room = cls(
            name=data.get("name", ""),
            room_type=RoomType(data.get("room_type", RoomType.OTHER.value))
        )
        
        # Load components
        for comp_data in data.get("components", []):
            room.add_component(
                component_type=ElectricalComponent(comp_data.get("type", ElectricalComponent.OTHER.value)),
                quantity=comp_data.get("quantity", 1),
                notes=comp_data.get("notes", ""),
                unit_price=comp_data.get("unit_price", 0),
                labor_hours=comp_data.get("labor_hours", 0)
            )
        
        # Load other properties
        room.notes = data.get("notes", "")
        room.measurements = data.get("measurements", {"length": 0, "width": 0, "height": 0})
        room.images = data.get("images", [])
        
        return room

class CustomerSelectionPage(CustomWizardPage):
    """Wizard page for selecting a customer."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the customer selection page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.customer_repository = ServiceLocator.get(CustomerRepository)
        self.customers: List[Customer] = []
        self.selected_customer: Optional[Customer] = None
        
        # Set up the rest of the UI
        self.setup_customer_ui()
        
        # Load customers
        self.load_customers()
    
    def setup_customer_ui(self):
        """Set up the customer selection UI."""
        form_layout = QFormLayout()
        
        # Customer selection
        self.customer_combo = QComboBox()
        self.customer_combo.setMinimumWidth(300)
        self.customer_combo.currentIndexChanged.connect(self.on_customer_selected)
        form_layout.addRow("Select Customer:", self.customer_combo)
        
        # Customer details
        self.customer_details = QTextEdit()
        self.customer_details.setReadOnly(True)
        self.customer_details.setMinimumHeight(150)
        form_layout.addRow("Customer Details:", self.customer_details)
        
        # Add buttons for manage customers
        buttons_layout = QHBoxLayout()
        
        self.new_customer_button = QPushButton("New Customer")
        self.new_customer_button.clicked.connect(self.on_new_customer)
        
        self.edit_customer_button = QPushButton("Edit Customer")
        self.edit_customer_button.clicked.connect(self.on_edit_customer)
        self.edit_customer_button.setEnabled(False)
        
        buttons_layout.addWidget(self.new_customer_button)
        buttons_layout.addWidget(self.edit_customer_button)
        buttons_layout.addStretch()
        
        # Add layouts to content
        self.content_layout.addLayout(form_layout)
        self.content_layout.addLayout(buttons_layout)
        self.content_layout.addStretch()
    
    async def load_customers(self):
        """Load customers from the repository."""
        try:
            # Show progress
            self.update_progress(10, "Loading customers...")
            
            # Get customers from repository
            self.customers = await self.customer_repository.list_customers(limit=100, offset=0)
            
            # Update progress
            self.update_progress(50, "Processing customer data...")
            
            # Update combo box
            self.customer_combo.clear()
            for customer in self.customers:
                self.customer_combo.addItem(f"{customer.name} ({customer.phone})", customer.customer_id)
            
            # Complete progress
            self.update_progress(100, "Customers loaded")
            
            # Hide progress bar after a delay
            QTimer.singleShot(2000, lambda: self.progress_widget.setVisible(False))
            
            # Mark step as complete if we have customers
            if self.customers:
                self.wizard_data.state = WizardStep.COMPLETED
                self.completeChanged.emit()
        except Exception as e:
            logger.error(f"Error loading customers: {e}", exc_info=True)
            self.show_error(f"Error loading customers: {str(e)}")
    
    def on_customer_selected(self, index: int):
        """
        Handle customer selection.
        
        Args:
            index: Selected index
        """
        if index < 0 or index >= len(self.customers):
            self.selected_customer = None
            self.customer_details.clear()
            self.edit_customer_button.setEnabled(False)
            return
        
        # Get selected customer
        self.selected_customer = self.customers[index]
        
        # Update details display
        details_html = f"""
        <h3>{self.selected_customer.name}</h3>
        <p><b>Phone:</b> {self.selected_customer.phone}</p>
        """
        
        if self.selected_customer.email:
            details_html += f"<p><b>Email:</b> {self.selected_customer.email}</p>"
        
        details_html += f"""
        <p><b>Address:</b><br>{self.selected_customer.billing_address.replace('\n', '<br>')}</p>
        <p><b>Customer Type:</b> {self.selected_customer.customer_type}</p>
        """
        
        if hasattr(self.selected_customer, 'property_details') and self.selected_customer.property_details:
            details_html += f"""
            <p><b>Property Address:</b><br>
            {self.selected_customer.property_details.address.replace('\n', '<br>')}</p>
            """
        
        self.customer_details.setHtml(details_html)
        
        # Enable edit button
        self.edit_customer_button.setEnabled(True)
        
        # Store customer ID in wizard data
        self.wizard_data.data["customer_id"] = self.selected_customer.customer_id
        self.wizard_data.data["customer"] = self.selected_customer
        
        # Mark step as complete
        self.wizard_data.state = WizardStep.COMPLETED
        self.completeChanged.emit()
    
    def on_new_customer(self):
        """Handle new customer button click."""
        # This would typically open the customer creation dialog
        # For now, just show a message
        QMessageBox.information(self, "New Customer", 
                             "This would open the new customer dialog.\nFor this example, please select an existing customer.")
    
    def on_edit_customer(self):
        """Handle edit customer button click."""
        # This would typically open the customer edit dialog
        # For now, just show a message
        if self.selected_customer:
            QMessageBox.information(self, "Edit Customer", 
                                 f"This would open the edit dialog for {self.selected_customer.name}.")
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if a customer is selected or the step is skipped
        return (self.selected_customer is not None) or super().isComplete()

class ProjectInfoPage(CustomWizardPage):
    """Wizard page for entering project information."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the project info page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.setup_project_ui()
    
    def setup_project_ui(self):
        """Set up the project information UI."""
        form_layout = QFormLayout()
        
        # Project title
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter a title for this estimate")
        self.title_edit.textChanged.connect(self.update_completion_status)
        form_layout.addRow("Estimate Title:", self.title_edit)
        
        # Project description
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Enter a detailed description of the project")
        self.description_edit.textChanged.connect(self.update_completion_status)
        form_layout.addRow("Description:", self.description_edit)
        
        # Estimate type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Room-by-Room Estimate", "Service Call", "Panel Upgrade", "New Construction", "Renovation", "Other"])
        form_layout.addRow("Estimate Type:", self.type_combo)
        
        # Add layout to content
        self.content_layout.addLayout(form_layout)
        self.content_layout.addStretch()
        
        # Load data if available
        if "title" in self.wizard_data.data:
            self.title_edit.setText(self.wizard_data.data["title"])
            
        if "description" in self.wizard_data.data:
            self.description_edit.setText(self.wizard_data.data["description"])
            
        if "type" in self.wizard_data.data:
            index = self.type_combo.findText(self.wizard_data.data["type"])
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
    
    def update_completion_status(self):
        """Update the page completion status based on inputs."""
        # Check if required fields are filled
        title = self.title_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        
        is_complete = bool(title and description)
        
        # Update wizard data
        self.wizard_data.data["title"] = title
        self.wizard_data.data["description"] = description
        self.wizard_data.data["type"] = self.type_combo.currentText()
        
        if is_complete:
            self.wizard_data.state = WizardStep.COMPLETED
        else:
            self.wizard_data.state = WizardStep.IN_PROGRESS
        
        # Update completion status
        self.completeChanged.emit()
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if title and description are provided or the step is skipped
        title = self.title_edit.text().strip()
        description = self.description_edit.toPlainText().strip()
        
        return (bool(title and description)) or super().isComplete()

class RoomListPage(CustomWizardPage):
    """Wizard page for managing the list of rooms to estimate."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the room list page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.rooms: List[Room] = []
        self.setup_room_list_ui()
        
        # Load rooms if available
        if "rooms" in self.wizard_data.data:
            self.load_rooms(self.wizard_data.data["rooms"])
    
    def setup_room_list_ui(self):
        """Set up the room list UI."""
        main_layout = QHBoxLayout()
        
        # Room list on the left
        list_layout = QVBoxLayout()
        
        list_layout.addWidget(QLabel("Rooms:"))
        
        self.room_list = QListWidget()
        self.room_list.setMinimumWidth(200)
        self.room_list.currentItemChanged.connect(self.on_room_selected)
        list_layout.addWidget(self.room_list)
        
        # Buttons for managing rooms
        buttons_layout = QHBoxLayout()
        
        self.add_room_button = QPushButton("Add Room")
        self.add_room_button.clicked.connect(self.on_add_room)
        
        self.edit_room_button = QPushButton("Edit Room")
        self.edit_room_button.clicked.connect(self.on_edit_room)
        self.edit_room_button.setEnabled(False)
        
        self.remove_room_button = QPushButton("Remove Room")
        self.remove_room_button.clicked.connect(self.on_remove_room)
        self.remove_room_button.setEnabled(False)
        
        buttons_layout.addWidget(self.add_room_button)
        buttons_layout.addWidget(self.edit_room_button)
        buttons_layout.addWidget(self.remove_room_button)
        
        list_layout.addLayout(buttons_layout)
        
        # Room details on the right
        details_layout = QVBoxLayout()
        
        details_layout.addWidget(QLabel("Room Details:"))
        
        self.details_widget = QWidget()
        self.details_layout = QFormLayout(self.details_widget)
        
        # Room name
        self.room_name_edit = QLineEdit()
        self.room_name_edit.setEnabled(False)
        self.details_layout.addRow("Name:", self.room_name_edit)
        
        # Room type
        self.room_type_combo = QComboBox()
        self.room_type_combo.addItems([rt.value for rt in RoomType])
        self.room_type_combo.setEnabled(False)
        self.details_layout.addRow("Type:", self.room_type_combo)
        
        # Room dimensions
        dimensions_layout = QHBoxLayout()
        
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0, 1000)
        self.length_spin.setSuffix(" ft")
        self.length_spin.setEnabled(False)
        
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0, 1000)
        self.width_spin.setSuffix(" ft")
        self.width_spin.setEnabled(False)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0, 20)
        self.height_spin.setSuffix(" ft")
        self.height_spin.setEnabled(False)
        
        dimensions_layout.addWidget(QLabel("L:"))
        dimensions_layout.addWidget(self.length_spin)
        dimensions_layout.addWidget(QLabel("W:"))
        dimensions_layout.addWidget(self.width_spin)
        dimensions_layout.addWidget(QLabel("H:"))
        dimensions_layout.addWidget(self.height_spin)
        
        self.details_layout.addRow("Dimensions:", dimensions_layout)
        
        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setEnabled(False)
        self.details_layout.addRow("Notes:", self.notes_edit)
        
        # Components list
        self.components_label = QLabel("Room has 0 components")
        self.details_layout.addRow("Components:", self.components_label)
        
        # Images count
        self.images_label = QLabel("Room has 0 images")
        self.details_layout.addRow("Images:", self.images_label)
        
        # Add details widget to layout
        details_layout.addWidget(self.details_widget)
        
        # Buttons for detailed editing
        detail_buttons_layout = QHBoxLayout()
        
        self.edit_components_button = QPushButton("Edit Components")
        self.edit_components_button.clicked.connect(self.on_edit_components)
        self.edit_components_button.setEnabled(False)
        
        self.manage_images_button = QPushButton("Manage Images")
        self.manage_images_button.clicked.connect(self.on_manage_images)
        self.manage_images_button.setEnabled(False)
        
        detail_buttons_layout.addWidget(self.edit_components_button)
        detail_buttons_layout.addWidget(self.manage_images_button)
        detail_buttons_layout.addStretch()
        
        details_layout.addLayout(detail_buttons_layout)
        details_layout.addStretch()
        
        # Add layouts to main layout
        main_layout.addLayout(list_layout, 1)
        main_layout.addLayout(details_layout, 2)
        
        # Add main layout to content
        self.content_layout.addLayout(main_layout)
        
        # Bottom section with summary
        summary_layout = QHBoxLayout()
        
        self.summary_label = QLabel("No rooms added yet")
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()
        
        self.content_layout.addLayout(summary_layout)
    
    def load_rooms(self, room_data: List[Dict[str, Any]]):
        """
        Load rooms from saved data.
        
        Args:
            room_data: List of room dictionaries
        """
        self.rooms = [Room.from_dict(data) for data in room_data]
        self.update_room_list()
        self.update_summary()
    
    def update_room_list(self):
        """Update the room list widget."""
        self.room_list.clear()
        
        for room in self.rooms:
            item = QListWidgetItem(f"{room.name} ({room.room_type.value})")
            item.setData(Qt.UserRole, self.rooms.index(room))
            self.room_list.addItem(item)
    
    def update_summary(self):
        """Update the summary label."""
        total_components = sum(len(room.components) for room in self.rooms)
        
        if not self.rooms:
            self.summary_label.setText("No rooms added yet")
        else:
            self.summary_label.setText(
                f"{len(self.rooms)} room{'s' if len(self.rooms) != 1 else ''} with "
                f"{total_components} component{'s' if total_components != 1 else ''} total"
            )
        
        # Store rooms in wizard data
        self.wizard_data.data["rooms"] = [room.to_dict() for room in self.rooms]
        
        # Update completion status
        if self.rooms:
            self.wizard_data.state = WizardStep.COMPLETED
        else:
            self.wizard_data.state = WizardStep.IN_PROGRESS
        
        self.completeChanged.emit()
    
    def on_room_selected(self, current, previous):
        """
        Handle room selection.
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        if not current:
            # Disable buttons
            self.edit_room_button.setEnabled(False)
            self.remove_room_button.setEnabled(False)
            self.edit_components_button.setEnabled(False)
            self.manage_images_button.setEnabled(False)
            
            # Disable and clear detail fields
            self.room_name_edit.setEnabled(False)
            self.room_name_edit.clear()
            self.room_type_combo.setEnabled(False)
            self.room_type_combo.setCurrentIndex(0)
            self.length_spin.setEnabled(False)
            self.length_spin.setValue(0)
            self.width_spin.setEnabled(False)
            self.width_spin.setValue(0)
            self.height_spin.setEnabled(False)
            self.height_spin.setValue(0)
            self.notes_edit.setEnabled(False)
            self.notes_edit.clear()
            self.components_label.setText("Room has 0 components")
            self.images_label.setText("Room has 0 images")
            return
        
        # Get selected room
        room_index = current.data(Qt.UserRole)
        room = self.rooms[room_index]
        
        # Enable buttons
        self.edit_room_button.setEnabled(True)
        self.remove_room_button.setEnabled(True)
        self.edit_components_button.setEnabled(True)
        self.manage_images_button.setEnabled(True)
        
        # Update detail fields
        self.room_name_edit.setText(room.name)
        
        type_index = self.room_type_combo.findText(room.room_type.value)
        if type_index >= 0:
            self.room_type_combo.setCurrentIndex(type_index)
        
        self.length_spin.setValue(room.measurements.get("length", 0))
        self.width_spin.setValue(room.measurements.get("width", 0))
        self.height_spin.setValue(room.measurements.get("height", 0))
        self.notes_edit.setText(room.notes)
        
        # Update component and image counts
        self.components_label.setText(f"Room has {len(room.components)} component{'s' if len(room.components) != 1 else ''}")
        self.images_label.setText(f"Room has {len(room.images)} image{'s' if len(room.images) != 1 else ''}")
    
    def on_add_room(self):
        """Handle add room button click."""
        dialog = RoomEditDialog(self)
        
        if dialog.exec_():
            # Get room data
            room_data = dialog.get_room_data()
            
            # Create room
            room = Room(
                name=room_data["name"],
                room_type=RoomType(room_data["type"])
            )
            
            # Set room measurements
            room.measurements["length"] = room_data["length"]
            room.measurements["width"] = room_data["width"]
            room.measurements["height"] = room_data["height"]
            
            # Set room notes
            room.notes = room_data["notes"]
            
            # Add room to list
            self.rooms.append(room)
            
            # Update UI
            self.update_room_list()
            self.update_summary()
    
    def on_edit_room(self):
        """Handle edit room button click."""
        # Get selected room
        selected_item = self.room_list.currentItem()
        if not selected_item:
            return
        
        room_index = selected_item.data(Qt.UserRole)
        room = self.rooms[room_index]
        
        # Open edit dialog
        dialog = RoomEditDialog(self, room)
        
        if dialog.exec_():
            # Get updated room data
            room_data = dialog.get_room_data()
            
            # Update room
            room.name = room_data["name"]
            room.room_type = RoomType(room_data["type"])
            room.measurements["length"] = room_data["length"]
            room.measurements["width"] = room_data["width"]
            room.measurements["height"] = room_data["height"]
            room.notes = room_data["notes"]
            
            # Update UI
            self.update_room_list()
            self.on_room_selected(selected_item, None)
            self.update_summary()
    
    def on_remove_room(self):
        """Handle remove room button click."""
        # Get selected room
        selected_item = self.room_list.currentItem()
        if not selected_item:
            return
        
        room_index = selected_item.data(Qt.UserRole)
        room = self.rooms[room_index]
        
        # Confirm removal
        result = QMessageBox.question(
            self,
            "Remove Room",
            f"Are you sure you want to remove the room '{room.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove room
            self.rooms.pop(room_index)
            
            # Update UI
            self.update_room_list()
            self.update_summary()
    
    def on_edit_components(self):
        """Handle edit components button click."""
        # Get selected room
        selected_item = self.room_list.currentItem()
        if not selected_item:
            return
        
        room_index = selected_item.data(Qt.UserRole)
        room = self.rooms[room_index]
        
        # Open components dialog
        dialog = ComponentsEditDialog(self, room)
        
        if dialog.exec_():
            # Get updated components
            room.components = dialog.get_components()
            
            # Update UI
            self.on_room_selected(selected_item, None)
            self.update_summary()
    
    def on_manage_images(self):
        """Handle manage images button click."""
        # Get selected room
        selected_item = self.room_list.currentItem()
        if not selected_item:
            return
        
        room_index = selected_item.data(Qt.UserRole)
        room = self.rooms[room_index]
        
        # Open images dialog
        dialog = ImagesManageDialog(self, room)
        
        if dialog.exec_():
            # Get updated images
            room.images = dialog.get_images()
            
            # Update UI
            self.on_room_selected(selected_item, None)
            self.update_summary()
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if there's at least one room or the step is skipped
        return bool(self.rooms) or super().isComplete()

class SummaryPage(CustomWizardPage):
    """Wizard page for showing the summary and generating the estimate."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the summary page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.estimate_repository = ServiceLocator.get(EstimateRepository)
        self.estimate_generator = ServiceLocator.get(EstimateGeneratorService)
        self.generated_estimate = None
        self.setup_summary_ui()
    
    def setup_summary_ui(self):
        """Set up the summary UI."""
        # Main layout
        layout = QVBoxLayout()
        
        # Summary title
        title_label = QLabel("Estimate Summary")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title_label)
        
        # Create tabs for different summary views
        self.tabs = QTabWidget()
        
        # Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Customer info section
        customer_group = QGroupBox("Customer Information")
        customer_layout = QFormLayout(customer_group)
        
        self.customer_name_label = QLabel()
        customer_layout.addRow("Name:", self.customer_name_label)
        
        self.customer_phone_label = QLabel()
        customer_layout.addRow("Phone:", self.customer_phone_label)
        
        self.customer_email_label = QLabel()
        customer_layout.addRow("Email:", self.customer_email_label)
        
        self.customer_address_label = QLabel()
        self.customer_address_label.setWordWrap(True)
        customer_layout.addRow("Address:", self.customer_address_label)
        
        overview_layout.addWidget(customer_group)
        
        # Project info section
        project_group = QGroupBox("Project Information")
        project_layout = QFormLayout(project_group)
        
        self.project_title_label = QLabel()
        project_layout.addRow("Title:", self.project_title_label)
        
        self.project_type_label = QLabel()
        project_layout.addRow("Type:", self.project_type_label)
        
        self.project_description_label = QLabel()
        self.project_description_label.setWordWrap(True)
        project_layout.addRow("Description:", self.project_description_label)
        
        overview_layout.addWidget(project_group)
        
        # Rooms summary section
        rooms_group = QGroupBox("Rooms Summary")
        rooms_layout = QVBoxLayout(rooms_group)
        
        self.rooms_list = QListWidget()
        self.rooms_list.setMaximumHeight(150)
        rooms_layout.addWidget(self.rooms_list)
        
        self.rooms_summary_label = QLabel()
        rooms_layout.addWidget(self.rooms_summary_label)
        
        overview_layout.addWidget(rooms_group)
        
        # Add overview tab
        self.tabs.addTab(overview_tab, "Overview")
        
        # Rooms tab (will be populated in initializePage)
        self.rooms_tab = QScrollArea()
        self.rooms_tab.setWidgetResizable(True)
        self.rooms_content = QWidget()
        self.rooms_layout = QVBoxLayout(self.rooms_content)
        self.rooms_tab.setWidget(self.rooms_content)
        
        self.tabs.addTab(self.rooms_tab, "Rooms Detail")
        
        # Estimate tab (will be populated after generation)
        self.estimate_tab = QScrollArea()
        self.estimate_tab.setWidgetResizable(True)
        self.estimate_content = QWidget()
        self.estimate_layout = QVBoxLayout(self.estimate_content)
        self.estimate_tab.setWidget(self.estimate_content)
        
        estimate_placeholder = QLabel("The estimate will appear here after generation.")
        estimate_placeholder.setAlignment(Qt.AlignCenter)
        self.estimate_layout.addWidget(estimate_placeholder)
        
        self.tabs.addTab(self.estimate_tab, "Estimate Preview")
        
        # Add tabs to layout
        layout.addWidget(self.tabs)
        
        # Generate estimate button
        self.generate_button = QPushButton("Generate Estimate")
        self.generate_button.setIcon(QIcon("icons/generate.png"))
        self.generate_button.clicked.connect(self.on_generate_estimate)
        
        layout.addWidget(self.generate_button)
        
        # Add layout to content
        self.content_layout.addLayout(layout)
    
    def _populate_customer_info(self):
        """Populate customer information in the UI."""
        # Get customer from wizard data
        customer = self.wizard_data.data.get("customer")
        if not customer:
            return
        
        # Update labels
        self.customer_name_label.setText(customer.name)
        self.customer_phone_label.setText(customer.phone)
        self.customer_email_label.setText(customer.email or "N/A")
        self.customer_address_label.setText(customer.billing_address.replace("\n", "<br>"))
    
    def _populate_project_info(self):
        """Populate project information in the UI."""
        # Get project info from wizard data
        title = self.wizard_data.data.get("title", "")
        description = self.wizard_data.data.get("description", "")
        project_type = self.wizard_data.data.get("type", "")
        
        # Update labels
        self.project_title_label.setText(title)
        self.project_type_label.setText(project_type)
        self.project_description_label.setText(description)
    
    def _populate_rooms_summary(self):
        """Populate rooms summary in the UI."""
        # Get rooms from wizard data
        rooms_data = self.wizard_data.data.get("rooms", [])
        rooms = [Room.from_dict(data) for data in rooms_data]
        
        # Update rooms list
        self.rooms_list.clear()
        for room in rooms:
            components_count = len(room.components)
            item = QListWidgetItem(f"{room.name} ({room.room_type.value}) - {components_count} component(s)")
            self.rooms_list.addItem(item)
        
        # Update summary label
        total_components = sum(len(room.components) for room in rooms)
        self.rooms_summary_label.setText(
            f"Total: {len(rooms)} room(s) with {total_components} component(s)"
        )
    
    def _populate_rooms_detail(self):
        """Populate detailed room information in the UI."""
        # Clear existing content
        while self.rooms_layout.count():
            item = self.rooms_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get rooms from wizard data
        rooms_data = self.wizard_data.data.get("rooms", [])
        rooms = [Room.from_dict(data) for data in rooms_data]
        
        # No rooms case
        if not rooms:
            no_rooms_label = QLabel("No rooms have been added.")
            no_rooms_label.setAlignment(Qt.AlignCenter)
            self.rooms_layout.addWidget(no_rooms_label)
            return
        
        # Add each room
        for room in rooms:
            room_group = QGroupBox(f"{room.name} ({room.room_type.value})")
            room_layout = QVBoxLayout(room_group)
            
            # Room dimensions
            dimensions_label = QLabel(
                f"Dimensions: {room.measurements.get('length', 0)} × "
                f"{room.measurements.get('width', 0)} × "
                f"{room.measurements.get('height', 0)} ft"
            )
            room_layout.addWidget(dimensions_label)
            
            # Room notes
            if room.notes:
                notes_label = QLabel(f"Notes: {room.notes}")
                notes_label.setWordWrap(True)
                room_layout.addWidget(notes_label)
            
            # Components table
            if room.components:
                components_label = QLabel(f"Components ({len(room.components)}):")
                room_layout.addWidget(components_label)
                
                table = QTableWidget(len(room.components), 4)
                table.setHorizontalHeaderLabels(["Component", "Quantity", "Unit Price", "Labor Hours"])
                table.setEditTriggers(QTableWidget.NoEditTriggers)
                table.setSelectionBehavior(QTableWidget.SelectRows)
                
                for i, component in enumerate(room.components):
                    table.setItem(i, 0, QTableWidgetItem(component["type"].value))
                    table.setItem(i, 1, QTableWidgetItem(str(component["quantity"])))
                    table.setItem(i, 2, QTableWidgetItem(f"${component['unit_price']:.2f}"))
                    table.setItem(i, 3, QTableWidgetItem(f"{component['labor_hours']:.1f}"))
                
                table.resizeColumnsToContents()
                table.setMaximumHeight(150)
                room_layout.addWidget(table)
            else:
                no_components_label = QLabel("No components added.")
                room_layout.addWidget(no_components_label)
            
            # Images count
            images_label = QLabel(f"Images: {len(room.images)}")
            room_layout.addWidget(images_label)
            
            # Add room group to layout
            self.rooms_layout.addWidget(room_group)
        
        # Add stretch at the end
        self.rooms_layout.addStretch()
    
    def initializePage(self):
        """Initialize the page content when the page is shown."""
        # Populate all sections
        self._populate_customer_info()
        self._populate_project_info()
        self._populate_rooms_summary()
        self._populate_rooms_detail()
        
        # Reset generated estimate
        self.generated_estimate = None
        
        # Reset estimate tab
        while self.estimate_layout.count():
            item = self.estimate_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        placeholder = QLabel("The estimate will appear here after generation.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.estimate_layout.addWidget(placeholder)
        
        # Enable generate button
        self.generate_button.setEnabled(True)
    
    async def on_generate_estimate(self):
        """Handle generate estimate button click."""
        try:
            # Disable button
            self.generate_button.setEnabled(False)
            
            # Show progress
            self.update_progress(10, "Preparing estimate data...")
            
            # Get customer
            customer = self.wizard_data.data.get("customer")
            if not customer:
                self.show_error("No customer selected")
                return
            
            # Get project info
            title = self.wizard_data.data.get("title", "")
            description = self.wizard_data.data.get("description", "")
            
            # Get rooms
            rooms_data = self.wizard_data.data.get("rooms", [])
            if not rooms_data:
                self.show_error("No rooms added")
                return
            
            # Update progress
            self.update_progress(20, "Creating estimate structure...")
            
            # Create customer info
            customer_info = CustomerInfo(
                name=customer.name,
                address=customer.billing_address,
                email=customer.email,
                phone=customer.phone
            )
            
            # Update progress
            self.update_progress(30, "Processing room data...")
            
            # Build room-by-room description
            estimate_description = f"{title}\n\n{description}\n\nRoom-by-Room Breakdown:\n"
            
            line_items = []
            for room_data in rooms_data:
                room = Room.from_dict(room_data)
                estimate_description += f"\n- {room.name} ({room.room_type.value}):"
                
                # Add room components as line items
                for component in room.components:
                    component_type = component["type"].value
                    quantity = component["quantity"]
                    unit_price = component["unit_price"]
                    labor_hours = component["labor_hours"]
                    
                    if quantity > 0:
                        # Add to description
                        estimate_description += f"\n  * {quantity} {component_type}"
                        
                        # Create line item
                        line_item = LineItem(
                            description=f"{component_type} - {room.name}",
                            quantity=quantity,
                            unit_price=unit_price,
                            labor_hours=labor_hours,
                            material_cost=unit_price * quantity,
                            measurement_source="Room-by-Room",
                            ai_confidence_score=1.0  # Manual entry has perfect confidence
                        )
                        
                        line_items.append(line_item)
            
            # Update progress
            self.update_progress(50, "Generating estimate...")
            
            # Get all image paths
            image_paths = []
            for room_data in rooms_data:
                image_paths.extend(room_data.get("images", []))
            
            # Generate estimate using AI service if images are available
            if image_paths:
                self.update_progress(60, "Processing images with AI...")
                
                try:
                    # Generate estimate using AI
                    ai_estimate = await self.estimate_generator.generate_estimate_from_media(
                        media_paths=image_paths,
                        customer_info=customer_info,
                        estimate_description=estimate_description,
                        progress_callback=lambda p, m: self.update_progress(60 + int(p * 0.3), m)
                    )
                    
                    # Merge manually created line items with AI-generated ones
                    for item in line_items:
                        ai_estimate.line_items.append(item)
                    
                    self.generated_estimate = ai_estimate
                    
                except Exception as e:
                    logger.error(f"Error generating AI estimate: {e}", exc_info=True)
                    self.update_progress(90, "Using manual estimate instead...")
                    
                    # Create basic estimate without AI
                    from models.estimate import Estimate, LaborCalculation, Material, StatusTracking
                    
                    # Calculate totals
                    labor_hours = sum(item.labor_hours * item.quantity for item in line_items)
                    hourly_rate = 75.0  # Default hourly rate
                    labor_cost = labor_hours * hourly_rate
                    material_cost = sum(item.material_cost for item in line_items)
                    total_amount = labor_cost + material_cost
                    
                    self.generated_estimate = Estimate(
                        customer=customer_info,
                        description=estimate_description,
                        line_items=line_items,
                        labor=LaborCalculation(
                            base_hours=labor_hours,
                            hourly_rate=hourly_rate,
                            complexity_factor=1.0
                        ),
                        materials=[],  # No detailed materials list for manual estimate
                        title=title
                    )
                    
                    self.generated_estimate.total_amount = total_amount
                    self.generated_estimate.labor_cost = labor_cost
                    self.generated_estimate.material_cost = material_cost
            else:
                # Create basic estimate without AI
                self.update_progress(80, "Creating manual estimate...")
                
                from models.estimate import Estimate, LaborCalculation, Material, StatusTracking
                
                # Calculate totals
                labor_hours = sum(item.labor_hours * item.quantity for item in line_items)
                hourly_rate = 75.0  # Default hourly rate
                labor_cost = labor_hours * hourly_rate
                material_cost = sum(item.material_cost for item in line_items)
                total_amount = labor_cost + material_cost
                
                self.generated_estimate = Estimate(
                    customer=customer_info,
                    description=estimate_description,
                    line_items=line_items,
                    labor=LaborCalculation(
                        base_hours=labor_hours,
                        hourly_rate=hourly_rate,
                        complexity_factor=1.0
                    ),
                    materials=[],  # No detailed materials list for manual estimate
                    title=title
                )
                
                self.generated_estimate.total_amount = total_amount
                self.generated_estimate.labor_cost = labor_cost
                self.generated_estimate.material_cost = material_cost
            
            # Update progress
            self.update_progress(90, "Updating UI...")
            
            # Populate estimate tab
            self._populate_estimate_tab()
            
            # Switch to estimate tab
            self.tabs.setCurrentWidget(self.estimate_tab)
            
            # Complete progress
            self.update_progress(100, "Estimate generated successfully")
            
            # Mark page as completed
            self.wizard_data.state = WizardStep.COMPLETED
            self.wizard_data.data["generated_estimate"] = self.generated_estimate
            self.completeChanged.emit()
            
            # Hide progress after delay
            QTimer.singleShot(2000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            logger.error(f"Error generating estimate: {e}", exc_info=True)
            self.show_error(f"Error generating estimate: {str(e)}")
            self.generate_button.setEnabled(True)
    
    def _populate_estimate_tab(self):
        """Populate the estimate preview tab with generated estimate."""
        # Clear existing content
        while self.estimate_layout.count():
            item = self.estimate_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.generated_estimate:
            placeholder = QLabel("No estimate has been generated yet.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.estimate_layout.addWidget(placeholder)
            return
        
        # Create estimate preview
        estimate = self.generated_estimate
        
        # Title
        title_label = QLabel(estimate.title or "Electrical Estimate")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.estimate_layout.addWidget(title_label)
        
        # Summary section
        summary_group = QGroupBox("Estimate Summary")
        summary_layout = QFormLayout(summary_group)
        
        summary_layout.addRow("Date:", QLabel(datetime.now().strftime("%Y-%m-%d")))
        summary_layout.addRow("Customer:", QLabel(estimate.customer.name))
        summary_layout.addRow("Total Amount:", QLabel(f"${estimate.total_amount:.2f}"))
        summary_layout.addRow("Labor Cost:", QLabel(f"${estimate.labor_cost:.2f}"))
        summary_layout.addRow("Material Cost:", QLabel(f"${estimate.material_cost:.2f}"))
        
        self.estimate_layout.addWidget(summary_group)
        
        # Description
        description_group = QGroupBox("Description")
        description_layout = QVBoxLayout(description_group)
        
        description_text = QTextEdit()
        description_text.setPlainText(estimate.description)
        description_text.setReadOnly(True)
        description_text.setMaximumHeight(150)
        description_layout.addWidget(description_text)
        
        self.estimate_layout.addWidget(description_group)
        
        # Line items
        items_group = QGroupBox("Line Items")
        items_layout = QVBoxLayout(items_group)
        
        if estimate.line_items:
            items_table = QTableWidget(len(estimate.line_items), 5)
            items_table.setHorizontalHeaderLabels(["Description", "Quantity", "Unit Price", "Labor Hours", "Total"])
            items_table.setEditTriggers(QTableWidget.NoEditTriggers)
            
            for i, item in enumerate(estimate.line_items):
                items_table.setItem(i, 0, QTableWidgetItem(item.description))
                items_table.setItem(i, 1, QTableWidgetItem(str(item.quantity)))
                items_table.setItem(i, 2, QTableWidgetItem(f"${item.unit_price:.2f}"))
                items_table.setItem(i, 3, QTableWidgetItem(f"{item.labor_hours:.1f}"))
                
                total = (item.unit_price * item.quantity) + (item.labor_hours * estimate.labor.hourly_rate * item.quantity)
                items_table.setItem(i, 4, QTableWidgetItem(f"${total:.2f}"))
            
            items_table.resizeColumnsToContents()
            items_layout.addWidget(items_table)
        else:
            items_layout.addWidget(QLabel("No line items"))
        
        self.estimate_layout.addWidget(items_group)
        
        # Labor calculation
        labor_group = QGroupBox("Labor Details")
        labor_layout = QFormLayout(labor_group)
        
        labor_layout.addRow("Total Hours:", QLabel(f"{estimate.labor.base_hours:.1f}"))
        labor_layout.addRow("Hourly Rate:", QLabel(f"${estimate.labor.hourly_rate:.2f}"))
        labor_layout.addRow("Complexity Factor:", QLabel(f"{estimate.labor.complexity_factor:.2f}"))
        
        if estimate.labor.notes:
            labor_layout.addRow("Notes:", QLabel(estimate.labor.notes))
        
        self.estimate_layout.addWidget(labor_group)
        
        # Materials
        if estimate.materials:
            materials_group = QGroupBox("Materials")
            materials_layout = QVBoxLayout(materials_group)
            
            materials_table = QTableWidget(len(estimate.materials), 3)
            materials_table.setHorizontalHeaderLabels(["Material", "Quantity", "Unit Cost"])
            materials_table.setEditTriggers(QTableWidget.NoEditTriggers)
            
            for i, material in enumerate(estimate.materials):
                materials_table.setItem(i, 0, QTableWidgetItem(material.name))
                materials_table.setItem(i, 1, QTableWidgetItem(str(material.quantity)))
                materials_table.setItem(i, 2, QTableWidgetItem(f"${material.unit_cost:.2f}"))
            
            materials_table.resizeColumnsToContents()
            materials_layout.addWidget(materials_table)
            
            self.estimate_layout.addWidget(materials_group)
        
        # Save button
        save_button = QPushButton("Save Estimate")
        save_button.setIcon(QIcon("icons/save.png"))
        save_button.clicked.connect(self.on_save_estimate)
        self.estimate_layout.addWidget(save_button)
        
        # Add stretch at the end
        self.estimate_layout.addStretch()
    
    async def on_save_estimate(self):
        """Handle save estimate button click."""
        if not self.generated_estimate:
            QMessageBox.warning(self, "Save Error", "No estimate to save.")
            return
        
        try:
            # Show progress
            self.update_progress(10, "Saving estimate...")
            
            # Create estimate in repository
            customer_id = self.wizard_data.data.get("customer").customer_id
            
            # Set status to draft
            self.generated_estimate.status_tracking.update_status("draft")
            
            estimate_id = await self.estimate_repository.create_estimate(
                self.generated_estimate, customer_id
            )
            
            if not estimate_id:
                self.show_error("Failed to save estimate")
                return
            
            # Update progress
            self.update_progress(100, "Estimate saved successfully")
            
            # Notify user
            QMessageBox.information(
                self,
                "Estimate Saved",
                f"Estimate saved with ID: {estimate_id}.\n\n"
                f"You can view and edit it in the Estimates tab."
            )
            
            # Hide progress after delay
            QTimer.singleShot(2000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            logger.error(f"Error saving estimate: {e}", exc_info=True)
            self.show_error(f"Error saving estimate: {str(e)}")
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if estimate is generated or the step is skipped
        return (self.generated_estimate is not None) or super().isComplete()

class RoomByRoomEstimateWizard(BaseWizard):
    """Wizard for room-by-room electrical estimation."""
    
    def __init__(self, parent=None):
        """
        Initialize the room-by-room estimation wizard.
        
        Args:
            parent: Parent widget
        """
        super().__init__("Room-by-Room Electrical Estimate", parent)
        
        # Configure wizard
        self.setOption(QWizard.HaveHelpButton, True)
        self.setPixmap(QWizard.LogoPixmap, QPixmap("icons/room_wizard_logo.png"))
        
        # Add steps
        self.add_customer_step()
        self.add_project_info_step()
        self.add_rooms_step()
        self.add_summary_step()
    
    def add_customer_step(self):
        """Add the customer selection step."""
        step_data = WizardStepData(
            step_id="customer",
            title="Select Customer",
            description="Select or create a customer for this estimate",
            optional=False,
            help_text="Choose an existing customer or create a new one. "
                     "Customer information will be included in the estimate."
        )
        
        self.add_step(step_data, CustomerSelectionPage)
    
    def add_project_info_step(self):
        """Add the project information step."""
        step_data = WizardStepData(
            step_id="project_info",
            title="Project Information",
            description="Enter basic information about the project",
            optional=False,
            help_text="Provide a title and description for the estimate. "
                     "Be as detailed as possible to ensure accurate pricing."
        )
        
        self.add_step(step_data, ProjectInfoPage)
    
    def add_rooms_step(self):
        """Add the rooms management step."""
        step_data = WizardStepData(
            step_id="rooms",
            title="Room-by-Room Breakdown",
            description="Add the rooms and electrical components for estimation",
            optional=False,
            help_text="Add each room in the project and specify the electrical components needed. "
                     "You can add measurements and upload images for each room."
        )
        
        self.add_step(step_data, RoomListPage)
    
    def add_summary_step(self):
        """Add the summary and estimate generation step."""
        step_data = WizardStepData(
            step_id="summary",
            title="Summary & Generate",
            description="Review your input and generate the estimate",
            optional=False,
            help_text="Review all the information you've provided and generate the estimate. "
                     "The system will calculate costs based on your input and any uploaded images."
        )
        
        self.add_step(step_data, SummaryPage)

# Helper dialogs for the wizard

class RoomEditDialog(QDialog):
    """Dialog for adding or editing a room."""
    
    def __init__(self, parent=None, room: Optional[Room] = None):
        """
        Initialize the room edit dialog.
        
        Args:
            parent: Parent widget
            room: Room to edit, or None for a new room
        """
        super().__init__(parent)
        self.room = room
        self.setWindowTitle("Room Details")
        self.setup_ui()
        
        # Load room data if provided
        if room:
            self.populate_room_data()
    
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # Room name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter room name")
        form_layout.addRow("Room Name:", self.name_edit)
        
        # Room type
        self.type_combo = QComboBox()
        self.type_combo.addItems([rt.value for rt in RoomType])
        form_layout.addRow("Room Type:", self.type_combo)
        
        # Room dimensions
        dimensions_layout = QHBoxLayout()
        
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0, 1000)
        self.length_spin.setSuffix(" ft")
        self.length_spin.setDecimals(1)
        
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0, 1000)
        self.width_spin.setSuffix(" ft")
        self.width_spin.setDecimals(1)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0, 20)
        self.height_spin.setSuffix(" ft")
        self.height_spin.setDecimals(1)
        self.height_spin.setValue(8.0)  # Default ceiling height
        
        dimensions_layout.addWidget(QLabel("L:"))
        dimensions_layout.addWidget(self.length_spin)
        dimensions_layout.addWidget(QLabel("W:"))
        dimensions_layout.addWidget(self.width_spin)
        dimensions_layout.addWidget(QLabel("H:"))
        dimensions_layout.addWidget(self.height_spin)
        
        form_layout.addRow("Dimensions:", dimensions_layout)
        
        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Enter any notes about this room")
        self.notes_edit.setMaximumHeight(100)
        form_layout.addRow("Notes:", self.notes_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def populate_room_data(self):
        """Populate the dialog with existing room data."""
        if not self.room:
            return
        
        self.name_edit.setText(self.room.name)
        
        type_index = self.type_combo.findText(self.room.room_type.value)
        if type_index >= 0:
            self.type_combo.setCurrentIndex(type_index)
        
        self.length_spin.setValue(self.room.measurements.get("length", 0))
        self.width_spin.setValue(self.room.measurements.get("width", 0))
        self.height_spin.setValue(self.room.measurements.get("height", 0))
        
        self.notes_edit.setText(self.room.notes)
    
    def get_room_data(self) -> Dict[str, Any]:
        """
        Get the room data from the dialog.
        
        Returns:
            Dictionary with room data
        """
        return {
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText(),
            "length": self.length_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "notes": self.notes_edit.toPlainText()
        }

class ComponentsEditDialog(QDialog):
    """Dialog for editing the components in a room."""
    
    def __init__(self, parent=None, room: Optional[Room] = None):
        """
        Initialize the components edit dialog.
        
        Args:
            parent: Parent widget
            room: Room containing the components
        """
        super().__init__(parent)
        self.room = room
        self.components = room.components.copy() if room else []
        
        self.setWindowTitle(f"Components - {room.name if room else 'Room'}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        self.setup_ui()
        self.update_components_list()
    
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Room info label
        if self.room:
            info_label = QLabel(f"Editing components for: {self.room.name} ({self.room.room_type.value})")
            layout.addWidget(info_label)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Components list on the left
        list_layout = QVBoxLayout()
        
        list_layout.addWidget(QLabel("Components:"))
        
        self.components_list = QListWidget()
        self.components_list.setMinimumWidth(250)
        self.components_list.currentItemChanged.connect(self.on_component_selected)
        list_layout.addWidget(self.components_list)
        
        # Buttons for managing components
        buttons_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.on_add_component)
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.on_edit_component)
        self.edit_button.setEnabled(False)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove_component)
        self.remove_button.setEnabled(False)
        
        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.edit_button)
        buttons_layout.addWidget(self.remove_button)
        
        list_layout.addLayout(buttons_layout)
        
        # Component details on the right
        details_layout = QVBoxLayout()
        
        details_layout.addWidget(QLabel("Component Details:"))
        
        self.details_widget = QWidget()
        self.details_layout = QFormLayout(self.details_widget)
        
        # Component type
        self.type_combo = QComboBox()
        self.type_combo.addItems([ec.value for ec in ElectricalComponent])
        self.details_layout.addRow("Type:", self.type_combo)
        
        # Quantity
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 1000)
        self.details_layout.addRow("Quantity:", self.quantity_spin)
        
        # Unit price
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0, 10000)
        self.price_spin.setPrefix("$")
        self.price_spin.setDecimals(2)
        self.details_layout.addRow("Unit Price:", self.price_spin)
        
        # Labor hours
        self.labor_spin = QDoubleSpinBox()
        self.labor_spin.setRange(0, 100)
        self.labor_spin.setSuffix(" hours")
        self.labor_spin.setDecimals(1)
        self.details_layout.addRow("Labor Hours:", self.labor_spin)
        
        # Notes
        self.notes_edit = QLineEdit()
        self.details_layout.addRow("Notes:", self.notes_edit)
        
        # Quick add buttons
        quick_add_layout = QGridLayout()
        quick_add_components = [
            ElectricalComponent.RECEPTACLE,
            ElectricalComponent.SWITCH,
            ElectricalComponent.LIGHT_FIXTURE,
            ElectricalComponent.GFCI_OUTLET,
            ElectricalComponent.CEILING_FAN,
            ElectricalComponent.DIMMER_SWITCH
        ]
        
        row, col = 0, 0
        for component in quick_add_components:
            button = QPushButton(f"+ {component.value.split(' ')[0]}")
            button.clicked.connect(lambda checked, c=component: self.quick_add_component(c))
            quick_add_layout.addWidget(button, row, col)
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        # Save and cancel buttons for details
        details_buttons_layout = QHBoxLayout()
        
        self.save_details_button = QPushButton("Add Component")
        self.save_details_button.clicked.connect(self.on_save_component_details)
        
        self.cancel_details_button = QPushButton("Cancel")
        self.cancel_details_button.clicked.connect(self.on_cancel_component_details)
        
        details_buttons_layout.addStretch()
        details_buttons_layout.addWidget(self.save_details_button)
        details_buttons_layout.addWidget(self.cancel_details_button)
        
        # Add everything to details layout
        details_layout.addWidget(self.details_widget)
        details_layout.addLayout(quick_add_layout)
        details_layout.addLayout(details_buttons_layout)
        details_layout.addStretch()
        
        # Add layouts to content layout
        content_layout.addLayout(list_layout, 1)
        content_layout.addLayout(details_layout, 1)
        
        # Add content layout to main layout
        layout.addLayout(content_layout)
        
        # Dialog buttons
        dialog_buttons_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        dialog_buttons_layout.addStretch()
        dialog_buttons_layout.addWidget(self.ok_button)
        dialog_buttons_layout.addWidget(self.cancel_button)
        
        layout.addLayout(dialog_buttons_layout)
        
        # Initially disable detail fields
        self.set_details_enabled(False)
    
    def update_components_list(self):
        """Update the components list widget."""
        self.components_list.clear()
        
        for comp in self.components:
            item = QListWidgetItem(f"{comp['quantity']} × {comp['type'].value}")
            item.setData(Qt.UserRole, self.components.index(comp))
            self.components_list.addItem(item)
    
    def set_details_enabled(self, enabled: bool):
        """
        Enable or disable component detail fields.
        
        Args:
            enabled: Whether fields should be enabled
        """
        self.type_combo.setEnabled(enabled)
        self.quantity_spin.setEnabled(enabled)
        self.price_spin.setEnabled(enabled)
        self.labor_spin.setEnabled(enabled)
        self.notes_edit.setEnabled(enabled)
        self.save_details_button.setEnabled(enabled)
        self.cancel_details_button.setEnabled(enabled)
    
    def on_component_selected(self, current, previous):
        """
        Handle component selection.
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        if not current:
            self.edit_button.setEnabled(False)
            self.remove_button.setEnabled(False)
            return
        
        self.edit_button.setEnabled(True)
        self.remove_button.setEnabled(True)
    
    def on_add_component(self):
        """Handle add component button click."""
        # Set default values
        self.type_combo.setCurrentIndex(0)
        self.quantity_spin.setValue(1)
        self.price_spin.setValue(0)
        self.labor_spin.setValue(0.5)  # Default labor time
        self.notes_edit.clear()
        
        # Set save button text
        self.save_details_button.setText("Add Component")
        
        # Enable details fields
        self.set_details_enabled(True)
    
    def on_edit_component(self):
        """Handle edit component button click."""
        # Get selected component
        selected_item = self.components_list.currentItem()
        if not selected_item:
            return
        
        comp_index = selected_item.data(Qt.UserRole)
        comp = self.components[comp_index]
        
        # Set values
        type_index = self.type_combo.findText(comp["type"].value)
        if type_index >= 0:
            self.type_combo.setCurrentIndex(type_index)
        
        self.quantity_spin.setValue(comp["quantity"])
        self.price_spin.setValue(comp["unit_price"])
        self.labor_spin.setValue(comp["labor_hours"])
        self.notes_edit.setText(comp["notes"])
        
        # Set save button text
        self.save_details_button.setText("Update Component")
        
        # Enable details fields
        self.set_details_enabled(True)
    
    def on_remove_component(self):
        """Handle remove component button click."""
        # Get selected component
        selected_item = self.components_list.currentItem()
        if not selected_item:
            return
        
        comp_index = selected_item.data(Qt.UserRole)
        comp = self.components[comp_index]
        
        # Confirm removal
        result = QMessageBox.question(
            self,
            "Remove Component",
            f"Are you sure you want to remove {comp['quantity']} × {comp['type'].value}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove component
            self.components.pop(comp_index)
            
            # Update UI
            self.update_components_list()
    
    def on_save_component_details(self):
        """Handle save component details button click."""
        # Get values
        comp_type = ElectricalComponent(self.type_combo.currentText())
        quantity = self.quantity_spin.value()
        unit_price = self.price_spin.value()
        labor_hours = self.labor_spin.value()
        notes = self.notes_edit.text()
        
        # Check if editing or adding
        if self.save_details_button.text() == "Update Component":
            # Get selected component
            selected_item = self.components_list.currentItem()
            if selected_item:
                comp_index = selected_item.data(Qt.UserRole)
                
                # Update component
                self.components[comp_index] = {
                    "type": comp_type,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "labor_hours": labor_hours,
                    "notes": notes
                }
        else:
            # Add new component
            self.components.append({
                "type": comp_type,
                "quantity": quantity,
                "unit_price": unit_price,
                "labor_hours": labor_hours,
                "notes": notes
            })
        
        # Update UI
        self.update_components_list()
        
        # Disable details fields
        self.set_details_enabled(False)
    
    def on_cancel_component_details(self):
        """Handle cancel component details button click."""
        # Disable details fields
        self.set_details_enabled(False)
    
    def quick_add_component(self, component_type: ElectricalComponent):
        """
        Quickly add a common component with default values.
        
        Args:
            component_type: Type of component to add
        """
        # Default values based on component type
        defaults = {
            ElectricalComponent.RECEPTACLE: {"price": 5.0, "labor": 0.5},
            ElectricalComponent.SWITCH: {"price": 5.0, "labor": 0.5},
            ElectricalComponent.LIGHT_FIXTURE: {"price": 50.0, "labor": 1.0},
            ElectricalComponent.GFCI_OUTLET: {"price": 15.0, "labor": 0.5},
            ElectricalComponent.CEILING_FAN: {"price": 150.0, "labor": 2.0},
            ElectricalComponent.DIMMER_SWITCH: {"price": 20.0, "labor": 0.5}
        }
        
        # Get defaults for this component
        default = defaults.get(component_type, {"price": 10.0, "labor": 0.5})
        
        # Add component
        self.components.append({
            "type": component_type,
            "quantity": 1,
            "unit_price": default["price"],
            "labor_hours": default["labor"],
            "notes": ""
        })
        
        # Update UI
        self.update_components_list()
    
    def get_components(self) -> List[Dict[str, Any]]:
        """
        Get the components from the dialog.
        
        Returns:
            List of component dictionaries
        """
        return self.components.copy()

class ImagesManageDialog(QDialog):
    """Dialog for managing images for a room."""
    
    def __init__(self, parent=None, room: Optional[Room] = None):
        """
        Initialize the images management dialog.
        
        Args:
            parent: Parent widget
            room: Room containing the images
        """
        super().__init__(parent)
        self.room = room
        self.images = room.images.copy() if room else []
        
        self.setWindowTitle(f"Images - {room.name if room else 'Room'}")
        self.setMinimumWidth(800)
        self.setMinimumHeight(500)
        
        self.setup_ui()
        self.update_images_list()
    
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Room info label
        if self.room:
            info_label = QLabel(f"Managing images for: {self.room.name} ({self.room.room_type.value})")
            layout.addWidget(info_label)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Images list on the left
        list_layout = QVBoxLayout()
        
        list_layout.addWidget(QLabel("Images:"))
        
        self.images_list = QListWidget()
        self.images_list.setMinimumWidth(200)
        self.images_list.currentItemChanged.connect(self.on_image_selected)
        list_layout.addWidget(self.images_list)
        
        # Buttons for managing images
        buttons_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Images")
        self.add_button.clicked.connect(self.on_add_images)
        
        self.remove_button = QPushButton("Remove Image")
        self.remove_button.clicked.connect(self.on_remove_image)
        self.remove_button.setEnabled(False)
        
        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.remove_button)
        
        list_layout.addLayout(buttons_layout)
        
        # Image preview on the right
        preview_layout = QVBoxLayout()
        
        preview_layout.addWidget(QLabel("Image Preview:"))
        
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setStyleSheet("border: 1px solid #cccccc;")
        preview_layout.addWidget(self.image_preview)
        
        # Image details
        self.image_details = QLabel("")
        preview_layout.addWidget(self.image_details)
        
        # Add layouts to content layout
        content_layout.addLayout(list_layout, 1)
        content_layout.addLayout(preview_layout, 2)
        
        # Add content layout to main layout
        layout.addLayout(content_layout)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def update_images_list(self):
        """Update the images list widget."""
        self.images_list.clear()
        
        for i, image_path in enumerate(self.images):
            # Get file name from path
            file_name = os.path.basename(image_path)
            
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, i)
            self.images_list.addItem(item)
    
    def on_image_selected(self, current, previous):
        """
        Handle image selection.
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        if not current:
            self.remove_button.setEnabled(False)
            self.image_preview.setText("No image selected")
            self.image_details.setText("")
            return
        
        self.remove_button.setEnabled(True)
        
        # Get selected image
        image_index = current.data(Qt.UserRole)
        image_path = self.images[image_index]
        
        # Load and display image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_preview.setText(f"Error loading image: {image_path}")
            return
        
        # Scale pixmap to fit the preview area while maintaining aspect ratio
        preview_size = self.image_preview.size()
        scaled_pixmap = pixmap.scaled(
            preview_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_preview.setPixmap(scaled_pixmap)
        
        # Show image details
        file_info = QFileInfo(image_path)
        self.image_details.setText(
            f"File: {file_info.fileName()}\n"
            f"Path: {file_info.filePath()}\n"
            f"Size: {pixmap.width()} × {pixmap.height()} pixels\n"
            f"File size: {file_info.size() / 1024:.1f} KB"
        )
    
    def on_add_images(self):
        """Handle add images button click."""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Images")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        if file_dialog.exec_():
            # Get selected files
            file_paths = file_dialog.selectedFiles()
            
            # Add to images list
            self.images.extend(file_paths)
            
            # Update UI
            self.update_images_list()
    
    def on_remove_image(self):
        """Handle remove image button click."""
        # Get selected image
        selected_item = self.images_list.currentItem()
        if not selected_item:
            return
        
        image_index = selected_item.data(Qt.UserRole)
        
        # Remove image
        self.images.pop(image_index)
        
        # Update UI
        self.update_images_list()
        
        # Clear preview if no images left
        if not self.images:
            self.image_preview.setText("No image selected")
            self.image_details.setText("")
    
    def get_images(self) -> List[str]:
        """
        Get the images from the dialog.
        
        Returns:
            List of image paths
        """
        return self.images.copy()
```

Now let's register our wizard with the wizard manager, which should be done in the application's main initialization:

### app.py (modification)
```python
# In the main() function, after initializing the UI:

# Register wizards
from ui.wizards.room_by_room_wizard import RoomByRoomEstimateWizard
from ui.wizards.wizard_base import wizard_manager

wizard_manager.register_wizard("room_by_room", RoomByRoomEstimateWizard)
```

## Let's also implement an improved progress dialog for better UX feedback:

### ui/dialogs/progress_dialog.py
```python
"""
Enhanced progress dialog with detailed status information and cancelation support.
"""

from typing import Optional, Callable, List, Dict, Any
import time
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
                           QPushButton, QDialogButtonBox, QScrollArea, QWidget, 
                           QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QObject, QThread
from PyQt5.QtGui import QIcon, QFont, QMovie

from utils.logger import logger
from utils.error_handling import ErrorHandler

class ProgressWorker(QObject):
    """Worker object for background operations with progress reporting."""
    
    # Signals for progress and completion
    progress = pyqtSignal(int, str)  # Progress percentage, status message
    finished = pyqtSignal(bool, object)  # Success flag, result
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, operation_func: Callable, args=None, kwargs=None):
        """
        Initialize the progress worker.
        
        Args:
            operation_func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        super().__init__()
        self.operation_func = operation_func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.is_canceled = False
    
    def run(self):
        """Execute the operation."""
        try:
            # Add progress callback to kwargs if function expects it
            if 'progress_callback' in self.operation_func.__code__.co_varnames:
                self.kwargs['progress_callback'] = self.report_progress
            
            # Execute the function
            result = self.operation_func(*self.args, **self.kwargs)
            
            # Check if operation was canceled
            if self.is_canceled:
                self.finished.emit(False, None)
            else:
                self.finished.emit(True, result)
                
        except Exception as e:
            # Log and emit error
            logger.error(f"Error in progress worker: {e}", exc_info=True)
            self.error.emit(str(e))
            self.finished.emit(False, None)
    
    def report_progress(self, percentage: int, message: str = ""):
        """
        Report progress to the dialog.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message
        """
        self.progress.emit(percentage, message)
    
    def cancel(self):
        """Cancel the operation."""
        self.is_canceled = True
        
        # If the operation function has a cancel method, call it
        if 'cancel' in self.kwargs:
            cancel_func = self.kwargs['cancel']
            if callable(cancel_func):
                try:
                    cancel_func()
                except Exception as e:
                    logger.error(f"Error canceling operation: {e}")

class AsyncProgressWorker(QObject):
    """Worker object for async background operations with progress reporting."""
    
    # Signals for progress and completion
    progress = pyqtSignal(int, str)  # Progress percentage, status message
    finished = pyqtSignal(bool, object)  # Success flag, result
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, async_operation_func: Callable, args=None, kwargs=None):
        """
        Initialize the async progress worker.
        
        Args:
            async_operation_func: Async function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        super().__init__()
        self.async_operation_func = async_operation_func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.is_canceled = False
    
    async def run(self):
        """Execute the async operation."""
        try:
            # Add progress callback to kwargs if function expects it
            if 'progress_callback' in self.async_operation_func.__code__.co_varnames:
                self.kwargs['progress_callback'] = self.report_progress
            
            # Execute the async function
            result = await self.async_operation_func(*self.args, **self.kwargs)
            
            # Check if operation was canceled
            if self.is_canceled:
                self.finished.emit(False, None)
            else:
                self.finished.emit(True, result)
                
        except Exception as e:
            # Log and emit error
            logger.error(f"Error in async progress worker: {e}", exc_info=True)
            self.error.emit(str(e))
            self.finished.emit(False, None)
    
    def report_progress(self, percentage: int, message: str = ""):
        """
        Report progress to the dialog.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message
        """
        self.progress.emit(percentage, message)
    
    def cancel(self):
        """Cancel the operation."""
        self.is_canceled = True
        
        # If the operation function has a cancel method, call it
        if 'cancel' in self.kwargs:
            cancel_func = self.kwargs['cancel']
            if callable(cancel_func):
                try:
                    cancel_func()
                except Exception as e:
                    logger.error(f"Error canceling operation: {e}")

class EnhancedProgressDialog(QDialog):
    """
    Enhanced progress dialog with detailed status information and cancelation support.
    """
    
    # Signal emitted when operation is completed or canceled
    operation_complete = pyqtSignal(bool, object)  # Success flag, result
    
    def __init__(self, title: str, description: str = "", parent=None, 
                 cancelable: bool = True, auto_close: bool = True,
                 show_details: bool = True, minimum_duration: int = 1000):
        """
        Initialize the progress dialog.
        
        Args:
            title: Dialog title
            description: Operation description
            parent: Parent widget
            cancelable: Whether the operation can be canceled
            auto_close: Whether to automatically close on completion
            show_details: Whether to show the detailed log
            minimum_duration: Minimum duration to show dialog in milliseconds
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.description = description
        self.cancelable = cancelable
        self.auto_close = auto_close
        self.show_details_default = show_details
        self.minimum_duration = minimum_duration
        self.start_time = 0
        
        # Worker thread and worker
        self.thread = None
        self.worker = None
        
        # Status log
        self.status_log: List[Dict[str, Any]] = []
        
        # Initialize UI
        self.setup_ui()
        
        # Connect signals for auto-close timer
        self.auto_close_timer = QTimer(self)
        self.auto_close_timer.setSingleShot(True)
        self.auto_close_timer.timeout.connect(self.check_auto_close)
    
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header section
        header_layout = QVBoxLayout()
        
        # Title label
        self.title_label = QLabel(self.description)
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.title_label.setWordWrap(True)
        header_layout.addWidget(self.title_label)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        header_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        header_layout.addWidget(self.progress_bar)
        
        # Time estimate
        self.time_label = QLabel("Estimating time remaining...")
        header_layout.addWidget(self.time_label)
        
        layout.addLayout(header_layout)
        
        # Details section
        self.details_frame = QFrame()
        details_layout = QVBoxLayout(self.details_frame)
        
        # Log display
        self.log_area = QScrollArea()
        self.log_area.setWidgetResizable(True)
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_area.setWidget(self.log_widget)
        self.log_area.setMinimumHeight(150)
        
        details_layout.addWidget(QLabel("Progress Log:"))
        details_layout.addWidget(self.log_area)
        
        # Initially hide details if configured
        self.details_frame.setVisible(self.show_details_default)
        
        layout.addWidget(self.details_frame)
        
        # Buttons section
        buttons_layout = QHBoxLayout()
        
        # Toggle details button
        self.details_button = QPushButton("Hide Details" if self.show_details_default else "Show Details")
        self.details_button.clicked.connect(self.toggle_details)
        buttons_layout.addWidget(self.details_button)
        
        buttons_layout.addStretch()
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(QIcon("icons/cancel.png"))
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.setEnabled(self.cancelable)
        self.cancel_button.setVisible(self.cancelable)
        buttons_layout.addWidget(self.cancel_button)
        
        layout.addLayout(buttons_layout)
        
        # Set dialog size
        self.resize(500, 300)
    
    def toggle_details(self):
        """Toggle the visibility of the details section."""
        self.details_frame.setVisible(not self.details_frame.isVisible())
        
        # Update button text
        if self.details_frame.isVisible():
            self.details_button.setText("Hide Details")
        else:
            self.details_button.setText("Show Details")
    
    def update_progress(self, percentage: int, message: str = ""):
        """
        Update the progress display.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message
        """
        # Ensure percentage is valid
        percentage = max(0, min(percentage, 100))
        
        # Update progress bar
        self.progress_bar.setValue(percentage)
        
        # Update status label if message is provided
        if message:
            self.status_label.setText(message)
            
            # Add to status log
            timestamp = time.time()
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            self.status_log.append({
                "timestamp": timestamp,
                "message": message,
                "percentage": percentage
            })
            
            # Add log entry
            log_entry = QLabel(f"{time_str} - {percentage}% - {message}")
            self.log_layout.addWidget(log_entry)
            
            # Scroll to bottom
            self.log_area.verticalScrollBar().setValue(
                self.log_area.verticalScrollBar().maximum()
            )
        
        # Update time estimate
        self.update_time_estimate(percentage)
        
        # Process events to ensure UI updates
        QApplication.processEvents()
    
    def update_time_estimate(self, percentage: int):
        """
        Update the time remaining estimate.
        
        Args:
            percentage: Current progress percentage
        """
        if percentage <= 0 or percentage >= 100:
            self.time_label.setText("")
            return
        
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time < 0.5 or percentage < 5:
            self.time_label.setText("Estimating time remaining...")
            return
        
        # Calculate estimated time remaining
        estimated_total_time = elapsed_time * 100 / percentage
        remaining_time = estimated_total_time - elapsed_time
        
        # Format time string
        if remaining_time < 60:
            time_str = f"About {int(remaining_time)} second{'s' if remaining_time != 1 else ''} remaining"
        elif remaining_time < 3600:
            minutes = int(remaining_time / 60)
            time_str = f"About {minutes} minute{'s' if minutes != 1 else ''} remaining"
        else:
            hours = int(remaining_time / 3600)
            minutes = int((remaining_time % 3600) / 60)
            time_str = f"About {hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''} remaining"
        
        self.time_label.setText(time_str)
    
    def run_operation(self, operation_func: Callable, *args, **kwargs):
        """
        Run an operation with progress tracking.
        
        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        """
        # Record start time
        self.start_time = time.time()
        
        # Create worker and thread
        self.worker = ProgressWorker(operation_func, args, kwargs)
        self.thread = QThread()
        
        # Move worker to thread
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.operation_finished)
        self.worker.error.connect(self.operation_error)
        
        # Start thread
        self.thread.start()
        
        # Show dialog
        self.exec_()
    
    async def run_async_operation(self, async_operation_func: Callable, *args, **kwargs):
        """
        Run an async operation with progress tracking.
        
        Args:
            async_operation_func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        """
        # Record start time
        self.start_time = time.time()
        
        # Create worker
        self.worker = AsyncProgressWorker(async_operation_func, args, kwargs)
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.operation_finished)
        self.worker.error.connect(self.operation_error)
        
        # Run the async operation
        await self.worker.run()
        
        # Show dialog
        self.exec_()
    
    def operation_finished(self, success: bool, result: Any):
        """
        Handle operation completion.
        
        Args:
            success: Whether the operation was successful
            result: Operation result
        """
        # Update progress
        if success:
            self.update_progress(100, "Operation completed successfully")
        else:
            self.update_progress(0, "Operation canceled")
        
        # Clean up thread if used
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        
        # Disable cancel button
        self.cancel_button.setEnabled(False)
        
        # Emit completion signal
        self.operation_complete.emit(success, result)
        
        # Start auto-close timer if enabled
        if self.auto_close and success:
            # Check if minimum duration has elapsed
            elapsed_time = (time.time() - self.start_time) * 1000  # Convert to ms
            remaining_time = max(0, self.minimum_duration - elapsed_time)
            
            # Start timer with remaining time
            self.auto_close_timer.start(int(remaining_time))
        else:
            # Change cancel button to close
            self.cancel_button.setText("Close")
            self.cancel_button.setIcon(QIcon("icons/close.png"))
            self.cancel_button.setEnabled(True)
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.accept)
    
    def operation_error(self, error_message: str):
        """
        Handle operation error.
        
        Args:
            error_message: Error message
        """
        # Update progress
        self.update_progress(0, f"Error: {error_message}")
        
        # Mark as error in UI
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff6666; }")
        
        # Clean up thread if used
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        
        # Change cancel button to close
        self.cancel_button.setText("Close")
        self.cancel_button.setIcon(QIcon("icons/close.png"))
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.reject)
        
        # Show details for error inspection
        if not self.details_frame.isVisible():
            self.toggle_details()
        
        # Emit completion signal
        self.operation_complete.emit(False, None)
    
    def cancel_operation(self):
        """Cancel the running operation."""
        # Update status
        self.status_label.setText("Canceling operation...")
        self.cancel_button.setEnabled(False)
        
        # Tell worker to cancel
        if self.worker:
            self.worker.cancel()
    
    def check_auto_close(self):
        """Check if the dialog should be auto-closed."""
        # Auto-close the dialog if enabled and operation was successful
        self.accept()
    
    def get_status_log(self) -> List[Dict[str, Any]]:
        """
        Get the status log entries.
        
        Returns:
            List of status log entries
        """
        return self.status_log.copy()
```

Now let's create a service panel upgrade wizard to demonstrate another workflow:

### ui/wizards/service_panel_wizard.py
```python
"""
Service panel upgrade wizard for guiding users through service panel estimation.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QListWidget, QListWidgetItem, QLineEdit, QComboBox, 
                           QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
                           QFileDialog, QMessageBox, QScrollArea, QGridLayout, QFrame,
                           QTabWidget, QTextEdit, QToolButton, QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont

from ui.wizards.wizard_base import BaseWizard, CustomWizardPage, WizardStepData
from models.customer import Customer
from models.estimate import Estimate, CustomerInfo, LineItem
from data.repositories.customer_repository import CustomerRepository
from data.repositories.estimate_repository import EstimateRepository
from services.estimate_generator import EstimateGeneratorService
from utils.logger import logger
from utils.error_handling import ErrorHandler
from utils.dependency_injection import ServiceLocator

# Reuse CustomerSelectionPage and ProjectInfoPage from room_by_room_wizard
from ui.wizards.room_by_room_wizard import CustomerSelectionPage, ProjectInfoPage, SummaryPage

class PanelType(Enum):
    """Types of electrical panels."""
    MAIN_PANEL = "Main Service Panel"
    SUB_PANEL = "Sub Panel"
    COMBO_PANEL = "Combination Panel"
    TRANSFER_SWITCH = "Transfer Switch Panel"
    METER_MAIN = "Meter-Main Panel"

class PanelSpecificationsPage(CustomWizardPage):
    """Wizard page for entering panel specifications."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the panel specifications page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.setup_panel_ui()
        
        # Load data if available
        if "panel_data" in self.wizard_data.data:
            self.load_panel_data(self.wizard_data.data["panel_data"])
    
    def setup_panel_ui(self):
        """Set up the panel specifications UI."""
        # Main layout using tabs for organization
        self.tabs = QTabWidget()
        
        # Panel basics tab
        basics_tab = QWidget()
        basics_layout = QFormLayout(basics_tab)
        
        # Panel type
        self.panel_type_combo = QComboBox()
        for panel_type in PanelType:
            self.panel_type_combo.addItem(panel_type.value)
        self.panel_type_combo.currentIndexChanged.connect(self.update_completion_status)
        basics_layout.addRow("Panel Type:", self.panel_type_combo)
        
        # Amperage
        self.amperage_combo = QComboBox()
        self.amperage_combo.addItems(["100A", "125A", "150A", "200A", "225A", "400A", "600A", "Other"])
        self.amperage_combo.currentIndexChanged.connect(self.update_amperage_options)
        basics_layout.addRow("Panel Size:", self.amperage_combo)
        
        # Custom amperage (shown only when "Other" is selected)
        self.custom_amperage_spin = QSpinBox()
        self.custom_amperage_spin.setRange(30, 1000)
        self.custom_amperage_spin.setSingleStep(5)
        self.custom_amperage_spin.setSuffix(" A")
        self.custom_amperage_spin.setVisible(False)
        basics_layout.addRow("Custom Amperage:", self.custom_amperage_spin)
        
        # Number of spaces
        self.spaces_spin = QSpinBox()
        self.spaces_spin.setRange(1, 200)
        self.spaces_spin.setValue(20)
        self.spaces_spin.setSingleStep(2)
        self.spaces_spin.valueChanged.connect(self.update_completion_status)
        basics_layout.addRow("Number of Spaces:", self.spaces_spin)
        
        # Voltage
        self.voltage_combo = QComboBox()
        self.voltage_combo.addItems(["120/240V Single Phase", "208/120V Three Phase", "480/277V Three Phase"])
        basics_layout.addRow("Voltage:", self.voltage_combo)
        
        # Manufacturer
        self.manufacturer_combo = QComboBox()
        self.manufacturer_combo.addItems(["Square D", "Eaton", "Siemens", "GE", "Leviton", "Other"])
        self.manufacturer_combo.setEditable(True)
        basics_layout.addRow("Manufacturer:", self.manufacturer_combo)
        
        # NEMA rating
        self.nema_combo = QComboBox()
        self.nema_combo.addItems(["NEMA 1 (Indoor)", "NEMA 3R (Outdoor)", "NEMA 4 (Water Tight)", "NEMA 12 (Dust Resistant)"])
        basics_layout.addRow("NEMA Rating:", self.nema_combo)
        
        # Main breaker
        self.main_breaker_check = QCheckBox("Include main breaker")
        self.main_breaker_check.setChecked(True)
        basics_layout.addRow("Main Breaker:", self.main_breaker_check)
        
        # Notes
        self.panel_notes_edit = QTextEdit()
        self.panel_notes_edit.setMaximumHeight(100)
        self.panel_notes_edit.setPlaceholderText("Enter any additional notes about the panel")
        basics_layout.addRow("Notes:", self.panel_notes_edit)
        
        self.tabs.addTab(basics_tab, "Panel Basics")
        
        # Location & Installation tab
        location_tab = QWidget()
        location_layout = QFormLayout(location_tab)
        
        # Installation type
        self.installation_group = QButtonGroup(self)
        installation_layout = QVBoxLayout()
        
        self.new_install_radio = QRadioButton("New Installation")
        self.upgrade_radio = QRadioButton("Upgrade Existing Panel")
        self.relocation_radio = QRadioButton("Panel Relocation")
        
        self.installation_group.addButton(self.new_install_radio, 1)
        self.installation_group.addButton(self.upgrade_radio, 2)
        self.installation_group.addButton(self.relocation_radio, 3)
        
        # Set default
        self.upgrade_radio.setChecked(True)
        
        installation_layout.addWidget(self.new_install_radio)
        installation_layout.addWidget(self.upgrade_radio)
        installation_layout.addWidget(self.relocation_radio)
        
        location_layout.addRow("Installation Type:", installation_layout)
        
        # Location
        self.location_combo = QComboBox()
        self.location_combo.addItems(["Basement", "Garage", "Utility Room", "Outdoor", "Kitchen", "Hallway", "Other"])
        self.location_combo.setEditable(True)
        location_layout.addRow("Panel Location:", self.location_combo)
        
        # Accessibility options
        accessibility_layout = QVBoxLayout()
        
        self.height_requirement_check = QCheckBox("Height Requirement (69-78 inches to center)")
        self.clearance_requirement_check = QCheckBox("Working Space Clearance (30 inch width, 36 inch depth)")
        self.dedicated_lighting_check = QCheckBox("Dedicated Lighting Needed")
        
        accessibility_layout.addWidget(self.height_requirement_check)
        accessibility_layout.addWidget(self.clearance_requirement_check)
        accessibility_layout.addWidget(self.dedicated_lighting_check)
        
        location_layout.addRow("Accessibility:", accessibility_layout)
        
        # Grounding
        self.grounding_combo = QComboBox()
        self.grounding_combo.addItems([
            "New ground rod(s) required",
            "Connect to existing grounding system",
            "Water pipe + ground rod connection",
            "Ufer (concrete-encased) ground",
            "Other grounding method"
        ])
        location_layout.addRow("Grounding:", self.grounding_combo)
        
        # Conduit/Cable Feed
        self.feed_combo = QComboBox()
        self.feed_combo.addItems([
            "Service entrance cable (overhead)",
            "Underground service lateral",
            "EMT conduit",
            "PVC conduit",
            "Rigid metal conduit",
            "Flexible metal conduit"
        ])
        location_layout.addRow("Service Feed:", self.feed_combo)
        
        # Main disconnect type
        self.disconnect_combo = QComboBox()
        self.disconnect_combo.addItems([
            "Built-in main breaker",
            "External main disconnect",
            "Multiple disconnects",
            "No main disconnect (sub-panel)"
        ])
        location_layout.addRow("Main Disconnect:", self.disconnect_combo)
        
        # Location notes
        self.location_notes_edit = QTextEdit()
        self.location_notes_edit.setPlaceholderText("Enter any additional details about location or installation requirements")
        self.location_notes_edit.setMaximumHeight(100)
        location_layout.addRow("Location Notes:", self.location_notes_edit)
        
        self.tabs.addTab(location_tab, "Location & Installation")
        
        # Circuits tab
        circuits_tab = QWidget()
        circuits_layout = QVBoxLayout(circuits_tab)
        
        # Existing circuits section
        existing_group = QGroupBox("Existing Circuits to Transfer")
        existing_layout = QVBoxLayout(existing_group)
        
        # Grid layout for common circuit types
        grid_layout = QGridLayout()
        
        # Row 1
        self.lighting_spin = QSpinBox()
        self.lighting_spin.setRange(0, 50)
        grid_layout.addWidget(QLabel("Lighting Circuits:"), 0, 0)
        grid_layout.addWidget(self.lighting_spin, 0, 1)
        
        self.receptacle_spin = QSpinBox()
        self.receptacle_spin.setRange(0, 50)
        grid_layout.addWidget(QLabel("Receptacle Circuits:"), 0, 2)
        grid_layout.addWidget(self.receptacle_spin, 0, 3)
        
        # Row 2
        self.gfci_spin = QSpinBox()
        self.gfci_spin.setRange(0, 20)
        grid_layout.addWidget(QLabel("GFCI Circuits:"), 1, 0)
        grid_layout.addWidget(self.gfci_spin, 1, 1)
        
        self.afci_spin = QSpinBox()
        self.afci_spin.setRange(0, 20)
        grid_layout.addWidget(QLabel("AFCI Circuits:"), 1, 2)
        grid_layout.addWidget(self.afci_spin, 1, 3)
        
        # Row 3
        self.dual_function_spin = QSpinBox()
        self.dual_function_spin.setRange(0, 20)
        grid_layout.addWidget(QLabel("Dual Function Circuits:"), 2, 0)
        grid_layout.addWidget(self.dual_function_spin, 2, 1)
        
        self.appliance_spin = QSpinBox()
        self.appliance_spin.setRange(0, 20)
        grid_layout.addWidget(QLabel("Appliance Circuits:"), 2, 2)
        grid_layout.addWidget(self.appliance_spin, 2, 3)
        
        # Row 4
        self.hvac_spin = QSpinBox()
        self.hvac_spin.setRange(0, 10)
        grid_layout.addWidget(QLabel("HVAC Circuits:"), 3, 0)
        grid_layout.addWidget(self.hvac_spin, 3, 1)
        
        self.other_spin = QSpinBox()
        self.other_spin.setRange(0, 30)
        grid_layout.addWidget(QLabel("Other Circuits:"), 3, 2)
        grid_layout.addWidget(self.other_spin, 3, 3)
        
        existing_layout.addLayout(grid_layout)
        
        # Special circuits
        special_layout = QFormLayout()
        
        self.two_pole_spin = QSpinBox()
        self.two_pole_spin.setRange(0, 20)
        special_layout.addRow("240V (2-pole) Circuits:", self.two_pole_spin)
        
        self.multi_pole_spin = QSpinBox()
        self.multi_pole_spin.setRange(0, 10)
        special_layout.addRow("3-pole Circuits:", self.multi_pole_spin)
        
        existing_layout.addLayout(special_layout)
        
        # Circuit transfer notes
        self.circuit_notes_edit = QTextEdit()
        self.circuit_notes_edit.setPlaceholderText("Enter any special requirements for circuit transfers or identification")
        self.circuit_notes_edit.setMaximumHeight(80)
        existing_layout.addWidget(QLabel("Circuit Transfer Notes:"))
        existing_layout.addWidget(self.circuit_notes_edit)
        
        circuits_layout.addWidget(existing_group)
        
        # New circuits section
        new_group = QGroupBox("New Circuits to Add")
        new_layout = QFormLayout(new_group)
        
        self.new_circuits_spin = QSpinBox()
        self.new_circuits_spin.setRange(0, 50)
        new_layout.addRow("Number of New Circuits:", self.new_circuits_spin)
        
        self.new_circuits_notes_edit = QTextEdit()
        self.new_circuits_notes_edit.setPlaceholderText("Describe new circuits to be added")
        self.new_circuits_notes_edit.setMaximumHeight(80)
        new_layout.addRow("New Circuits Description:", self.new_circuits_notes_edit)
        
        circuits_layout.addWidget(new_group)
        
        # Surge protection
        self.surge_protection_check = QCheckBox("Add whole-house surge protection")
        circuits_layout.addWidget(self.surge_protection_check)
        
        self.tabs.addTab(circuits_tab, "Circuits")
        
        # Images tab
        images_tab = QWidget()
        images_layout = QVBoxLayout(images_tab)
        
        # Images list
        images_layout.addWidget(QLabel("Panel Images:"))
        
        self.images_list = QListWidget()
        images_layout.addWidget(self.images_list)
        
        # Image buttons
        image_buttons_layout = QHBoxLayout()
        
        self.add_images_button = QPushButton("Add Images")
        self.add_images_button.clicked.connect(self.on_add_images)
        
        self.remove_image_button = QPushButton("Remove Image")
        self.remove_image_button.clicked.connect(self.on_remove_image)
        self.remove_image_button.setEnabled(False)
        
        image_buttons_layout.addWidget(self.add_images_button)
        image_buttons_layout.addWidget(self.remove_image_button)
        image_buttons_layout.addStretch()
        
        images_layout.addLayout(image_buttons_layout)
        
        # Image preview
        images_layout.addWidget(QLabel("Image Preview:"))
        
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setStyleSheet("border: 1px solid #ccc;")
        images_layout.addWidget(self.image_preview)
        
        self.tabs.addTab(images_tab, "Images")
        
        # Connect images list selection
        self.images_list.currentItemChanged.connect(self.on_image_selected)
        
        # Add tabs to content layout
        self.content_layout.addWidget(self.tabs)
        
        # Initialize empty images list
        self.images = []
    
    def update_amperage_options(self, index: int):
        """
        Handle amperage selection changes.
        
        Args:
            index: Current index
        """
        # Show custom amperage field if "Other" is selected
        self.custom_amperage_spin.setVisible(self.amperage_combo.currentText() == "Other")
        
        # Update completion status
        self.update_completion_status()
    
    def update_completion_status(self):
        """Update the page completion status based on inputs."""
        # Check required fields
        has_amperage = (self.amperage_combo.currentText() != "Other" or 
                       self.custom_amperage_spin.value() > 0)
        has_spaces = self.spaces_spin.value() > 0
        
        is_complete = has_amperage and has_spaces
        
        # Store panel data
        self.store_panel_data()
        
        # Update wizard state
        if is_complete:
            self.wizard_data.state = WizardStep.COMPLETED
        else:
            self.wizard_data.state = WizardStep.IN_PROGRESS
        
        # Update completion status
        self.completeChanged.emit()
    
    def store_panel_data(self):
        """Store panel data in wizard data."""
        # Get amperage
        if self.amperage_combo.currentText() == "Other":
            amperage = self.custom_amperage_spin.value()
        else:
            amperage = int(self.amperage_combo.currentText().replace("A", ""))
        
        # Get installation type
        installation_type = "New Installation"
        if self.upgrade_radio.isChecked():
            installation_type = "Upgrade Existing Panel"
        elif self.relocation_radio.isChecked():
            installation_type = "Panel Relocation"
        
        # Build panel data
        panel_data = {
            "panel_type": self.panel_type_combo.currentText(),
            "amperage": amperage,
            "spaces": self.spaces_spin.value(),
            "voltage": self.voltage_combo.currentText(),
            "manufacturer": self.manufacturer_combo.currentText(),
            "nema_rating": self.nema_combo.currentText(),
            "main_breaker": self.main_breaker_check.isChecked(),
            "panel_notes": self.panel_notes_edit.toPlainText(),
            
            "installation_type": installation_type,
            "location": self.location_combo.currentText(),
            "height_requirement": self.height_requirement_check.isChecked(),
            "clearance_requirement": self.clearance_requirement_check.isChecked(),
            "dedicated_lighting": self.dedicated_lighting_check.isChecked(),
            "grounding": self.grounding_combo.currentText(),
            "feed": self.feed_combo.currentText(),
            "disconnect": self.disconnect_combo.currentText(),
            "location_notes": self.location_notes_edit.toPlainText(),
            
            "circuits": {
                "lighting": self.lighting_spin.value(),
                "receptacle": self.receptacle_spin.value(),
                "gfci": self.gfci_spin.value(),
                "afci": self.afci_spin.value(),
                "dual_function": self.dual_function_spin.value(),
                "appliance": self.appliance_spin.value(),
                "hvac": self.hvac_spin.value(),
                "other": self.other_spin.value(),
                "two_pole": self.two_pole_spin.value(),
                "multi_pole": self.multi_pole_spin.value()
            },
            "circuit_notes": self.circuit_notes_edit.toPlainText(),
            "new_circuits": self.new_circuits_spin.value(),
            "new_circuits_notes": self.new_circuits_notes_edit.toPlainText(),
            "surge_protection": self.surge_protection_check.isChecked(),
            
            "images": self.images
        }
        
        # Store in wizard data
        self.wizard_data.data["panel_data"] = panel_data
    
    def load_panel_data(self, panel_data: Dict[str, Any]):
        """
        Load panel data into UI.
        
        Args:
            panel_data: Panel data dictionary
        """
        # Panel basics
        panel_type_index = self.panel_type_combo.findText(panel_data.get("panel_type", ""))
        if panel_type_index >= 0:
            self.panel_type_combo.setCurrentIndex(panel_type_index)
        
        # Amperage
        amperage = panel_data.get("amperage", 200)
        amperage_str = f"{amperage}A"
        amperage_index = self.amperage_combo.findText(amperage_str)
        if amperage_index >= 0:
            self.amperage_combo.setCurrentIndex(amperage_index)
        else:
            # Set to "Other" and set custom value
            other_index = self.amperage_combo.findText("Other")
            if other_index >= 0:
                self.amperage_combo.setCurrentIndex(other_index)
                self.custom_amperage_spin.setValue(amperage)
        
        # Other basic fields
        self.spaces_spin.setValue(panel_data.get("spaces", 20))
        
        voltage_index = self.voltage_combo.findText(panel_data.get("voltage", ""), Qt.MatchContains)
        if voltage_index >= 0:
            self.voltage_combo.setCurrentIndex(voltage_index)
        
        manufacturer_index = self.manufacturer_combo.findText(panel_data.get("manufacturer", ""), Qt.MatchContains)
        if manufacturer_index >= 0:
            self.manufacturer_combo.setCurrentIndex(manufacturer_index)
        else:
            self.manufacturer_combo.setCurrentText(panel_data.get("manufacturer", ""))
        
        nema_index = self.nema_combo.findText(panel_data.get("nema_rating", ""), Qt.MatchContains)
        if nema_index >= 0:
            self.nema_combo.setCurrentIndex(nema_index)
        
        self.main_breaker_check.setChecked(panel_data.get("main_breaker", True))
        self.panel_notes_edit.setText(panel_data.get("panel_notes", ""))
        
        # Location & Installation
        installation_type = panel_data.get("installation_type", "Upgrade Existing Panel")
        if installation_type == "New Installation":
            self.new_install_radio.setChecked(True)
        elif installation_type == "Upgrade Existing Panel":
            self.upgrade_radio.setChecked(True)
        elif installation_type == "Panel Relocation":
            self.relocation_radio.setChecked(True)
        
        location_index = self.location_combo.findText(panel_data.get("location", ""), Qt.MatchContains)
        if location_index >= 0:
            self.location_combo.setCurrentIndex(location_index)
        else:
            self.location_combo.setCurrentText(panel_data.get("location", ""))
        
        self.height_requirement_check.setChecked(panel_data.get("height_requirement", False))
        self.clearance_requirement_check.setChecked(panel_data.get("clearance_requirement", False))
        self.dedicated_lighting_check.setChecked(panel_data.get("dedicated_lighting", False))
        
        grounding_index = self.grounding_combo.findText(panel_data.get("grounding", ""), Qt.MatchContains)
        if grounding_index >= 0:
            self.grounding_combo.setCurrentIndex(grounding_index)
        
        feed_index = self.feed_combo.findText(panel_data.get("feed", ""), Qt.MatchContains)
        if feed_index >= 0:
            self.feed_combo.setCurrentIndex(feed_index)
        
        disconnect_index = self.disconnect_combo.findText(panel_data.get("disconnect", ""), Qt.MatchContains)
        if disconnect_index >= 0:
            self.disconnect_combo.setCurrentIndex(disconnect_index)
        
        self.location_notes_edit.setText(panel_data.get("location_notes", ""))
        
        # Circuits
        circuits = panel_data.get("circuits", {})
        self.lighting_spin.setValue(circuits.get("lighting", 0))
        self.receptacle_spin.setValue(circuits.get("receptacle", 0))
        self.gfci_spin.setValue(circuits.get("gfci", 0))
        self.afci_spin.setValue(circuits.get("afci", 0))
        self.dual_function_spin.setValue(circuits.get("dual_function", 0))
        self.appliance_spin.setValue(circuits.get("appliance", 0))
        self.hvac_spin.setValue(circuits.get("hvac", 0))
        self.other_spin.setValue(circuits.get("other", 0))
        self.two_pole_spin.setValue(circuits.get("two_pole", 0))
        self.multi_pole_spin.setValue(circuits.get("multi_pole", 0))
        
        self.circuit_notes_edit.setText(panel_data.get("circuit_notes", ""))
        self.new_circuits_spin.setValue(panel_data.get("new_circuits", 0))
        self.new_circuits_notes_edit.setText(panel_data.get("new_circuits_notes", ""))
        self.surge_protection_check.setChecked(panel_data.get("surge_protection", False))
        
        # Images
        self.images = panel_data.get("images", [])
        self.update_images_list()
    
    def on_add_images(self):
        """Handle add images button click."""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select Images")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        if file_dialog.exec_():
            # Get selected files
            file_paths = file_dialog.selectedFiles()
            
            # Add to images list
            self.images.extend(file_paths)
            
            # Update UI
            self.update_images_list()
            
            # Update wizard data
            self.store_panel_data()
    
    def on_remove_image(self):
        """Handle remove image button click."""
        # Get selected image
        selected_item = self.images_list.currentItem()
        if not selected_item:
            return
        
        image_index = selected_item.data(Qt.UserRole)
        
        # Remove image
        self.images.pop(image_index)
        
        # Update UI
        self.update_images_list()
        
        # Clear preview if no images left
        if not self.images:
            self.image_preview.setText("No image selected")
        
        # Update wizard data
        self.store_panel_data()
    
    def update_images_list(self):
        """Update the images list widget."""
        self.images_list.clear()
        
        for i, image_path in enumerate(self.images):
            # Get file name from path
            file_name = os.path.basename(image_path)
            
            item = QListWidgetItem(file_name)
            item.setData(Qt.UserRole, i)
            self.images_list.addItem(item)
    
    def on_image_selected(self, current, previous):
        """
        Handle image selection.
        
        Args:
            current: Current selected item
            previous: Previously selected item
        """
        if not current:
            self.remove_image_button.setEnabled(False)
            self.image_preview.setText("No image selected")
            return
        
        self.remove_image_button.setEnabled(True)
        
        # Get selected image
        image_index = current.data(Qt.UserRole)
        image_path = self.images[image_index]
        
        # Load and display image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_preview.setText(f"Error loading image: {image_path}")
            return
        
        # Scale pixmap to fit the preview area while maintaining aspect ratio
        preview_size = self.image_preview.size()
        scaled_pixmap = pixmap.scaled(
            preview_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_preview.setPixmap(scaled_pixmap)
    
    def isComplete(self) -> bool:
        """
        Check if the page is complete.
        
        Returns:
            Whether the page is complete
        """
        # Page is complete if required fields are filled or the step is skipped
        has_amperage = (self.amperage_combo.currentText() != "Other" or 
                       self.custom_amperage_spin.value() > 0)
        has_spaces = self.spaces_spin.value() > 0
        
        return (has_amperage and has_spaces) or super().isComplete()

class CalculationsPage(CustomWizardPage):
    """Wizard page for displaying panel calculations and requirements."""
    
    def __init__(self, wizard_data: WizardStepData, parent=None):
        """
        Initialize the calculations page.
        
        Args:
            wizard_data: Wizard step data
            parent: Parent widget
        """
        super().__init__(wizard_data, parent)
        self.setup_calculations_ui()
    
    def setup_calculations_ui(self):
        """Set up the calculations UI."""
        main_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Service Panel Calculations")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Explanation text
        explanation_text = (
            "This page shows calculations and requirements for your panel upgrade. "
            "These calculations are based on the National Electrical Code (NEC) and "
            "help ensure your new panel meets load requirements."
        )
        explanation_label = QLabel(explanation_text)
        explanation_label.setWordWrap(True)
        main_layout.addWidget(explanation_label)
        
        # Calculations tabs
        self.calculations_tabs = QTabWidget()
        
        # Load calculation tab
        load_tab = QWidget()
        load_layout = QVBoxLayout(load_tab)
        
        # Load calculation viewer
        self.load_calc_text = QTextEdit()
        self.load_calc_text.setReadOnly(True)
        load_layout.addWidget(self.load_calc_text)
        
        self.calculations_tabs.addTab(load_tab, "Load Calculation")
        
        # Code requirements tab
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)
        
        self.code_requirements_text = QTextEdit()
        self.code_requirements_text.setReadOnly(True)
        code_layout.addWidget(self.code_requirements_text)
        
        self.calculations_tabs.addTab(code_tab, "Code Requirements")
        
        # Material requirements tab
        materials_tab = QWidget()
        materials_layout = QVBoxLayout(materials_tab)
        
        self.materials_text = QTextEdit()
        self.materials_text.setReadOnly(True)
        materials_layout.addWidget(self.materials_text)
        
        self.calculations_tabs.addTab(materials_tab, "Material Requirements")
        
        main_layout.addWidget(self.calculations_tabs, 1)
        
        # Calculation buttons
        buttons_layout = QHBoxLayout()
        
        self.calculate_button = QPushButton("Calculate Requirements")
        self.calculate_button.setIcon(QIcon("icons/calculate.png"))
        self.calculate_button.clicked.connect(self.on_calculate)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.calculate_button)
        buttons_layout.addStretch()
        
        main_layout.addLayout(buttons_layout)
        
        # Add main layout to content layout
        self.content_layout.addLayout(main_layout)
    
    def initializePage(self):
        """Initialize the page content when the page is shown."""
        # Run calculations automatically
        self.on_calculate()
    
    def on_calculate(self):
        """Handle calculate button click."""
        try:
            # Show progress
            self.update_progress(10, "Calculating panel requirements...")
            
            # Get panel data
            panel_data = self.wizard_data.data.get("panel_data", {})
            if not panel_data:
                self.show_error("No panel data available")
                return
            
            # Update progress
            self.update_progress(30, "Calculating load requirements...")
            
            # Perform load calculation
            load_calculation = self.calculate_load(panel_data)
            self.load_calc_text.setHtml(load_calculation)
            
            # Update progress
            self.update_progress(60, "Determining code requirements...")
            
            # Determine code requirements
            code_requirements = self.determine_code_requirements(panel_data)
            self.code_requirements_text.setHtml(code_requirements)
            
            # Update progress
            self.update_progress(80, "Listing material requirements...")
            
            # List material requirements
            material_requirements = self.list_material_requirements(panel_data)
            self.materials_text.setHtml(material_requirements)
            
            # Update progress
            self.update_progress(100, "Calculations complete")
            
            # Mark page as completed
            self.wizard_data.state = WizardStep.COMPLETED
            self.completeChanged.emit()
            
            # Hide progress after a delay
            QTimer.singleShot(2000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            logger.error(f"Error performing calculations: {e}", exc_info=True)
            self.show_error(f"Error performing calculations: {str(e)}")
    
    def calculate_load(self, panel_data: Dict[str, Any]) -> str:
        """
        Calculate electrical load for the panel.
        
        Args:
            panel_data: Panel specifications data
            
        Returns:
            HTML formatted load calculation
        """
        # Extract relevant data
        circuits = panel_data.get("circuits", {})
        amperage = panel_data.get("amperage", 200)
        
        # Calculate circuit loads
        lighting_circuits = circuits.get("lighting", 0)
        receptacle_circuits = circuits.get("receptacle", 0)
        gfci_circuits = circuits.get("gfci", 0)
        afci_circuits = circuits.get("afci", 0)
        dual_function_circuits = circuits.get("dual_function", 0)
        appliance_circuits = circuits.get("appliance", 0)
        hvac_circuits = circuits.get("hvac", 0)
        two_pole_circuits = circuits.get("two_pole", 0)
        
        # Standard load assumptions (VA)
        lighting_load = lighting_circuits * 1200  # 1200 VA per lighting circuit
        receptacle_load = receptacle_circuits * 1500  # 1500 VA per receptacle circuit
        gfci_load = gfci_circuits * 1500  # 1500 VA per GFCI circuit
        afci_load = afci_circuits * 1500  # 1500 VA per AFCI circuit
        dual_function_load = dual_function_circuits * 1500  # 1500 VA per dual function circuit
        appliance_load = appliance_circuits * 1500  # 1500 VA per general appliance circuit
        hvac_load = hvac_circuits * 3000  # 3000 VA per HVAC circuit
        two_pole_load = two_pole_circuits * 3000  # 3000 VA per 2-pole circuit
        
        # Calculate total load
        general_lighting_load = lighting_load + receptacle_load + gfci_load + afci_load + dual_function_load
        
        # Apply demand factors to general lighting load
        demand_factor = 1.0
        if general_lighting_load > 10000:
            # First 10kVA at 100%, remainder at 40%
            demand_factor = (10000 + (general_lighting_load - 10000) * 0.4) / general_lighting_load
        
        adjusted_general_load = general_lighting_load * demand_factor
        
        # Calculate total connected load
        total_connected_load = general_lighting_load + appliance_load + hvac_load + two_pole_load
        
        # Calculate total demand load
        total_demand_load = adjusted_general_load + appliance_load + hvac_load + two_pole_load
        
        # Calculate amps at specified voltage (assuming 240V)
        voltage = 240
        total_amps = total_demand_load / voltage
        
        # Determine if panel size is adequate
        is_adequate = amperage >= (total_amps * 1.25)  # 25% safety margin
        
        # Format the results as HTML
        result = f"""
        <h3>Load Calculation Summary</h3>
        
        <h4>Circuit Load Breakdown</h4>
        <ul>
            <li>Lighting Circuits: {lighting_circuits} × 1200 VA = {lighting_load} VA</li>
            <li>Receptacle Circuits: {receptacle_circuits} × 1500 VA = {receptacle_load} VA</li>
            <li>GFCI Circuits: {gfci_circuits} × 1500 VA = {gfci_load} VA</li>
            <li>AFCI Circuits: {afci_circuits} × 1500 VA = {afci_load} VA</li>
            <li>Dual Function Circuits: {dual_function_circuits} × 1500 VA = {dual_function_load} VA</li>
            <li>Appliance Circuits: {appliance_circuits} × 1500 VA = {appliance_load} VA</li>
            <li>HVAC Circuits: {hvac_circuits} × 3000 VA = {hvac_load} VA</li>
            <li>240V (2-pole) Circuits: {two_pole_circuits} × 3000 VA = {two_pole_load} VA</li>
        </ul>
        
        <h4>Load Calculations</h4>
        <ul>
            <li>General Lighting Load: {general_lighting_load} VA</li>
            <li>Demand Factor Applied: {demand_factor:.2f}</li>
            <li>Adjusted General Load: {adjusted_general_load:.2f} VA</li>
            <li>Total Connected Load: {total_connected_load} VA</li>
            <li>Total Demand Load: {total_demand_load:.2f} VA</li>
        </ul>
        
        <h4>Amperage Requirements</h4>
        <ul>
            <li>Calculated Load: {total_amps:.2f} Amps at {voltage}V</li>
            <li>With 25% Safety Margin: {total_amps * 1.25:.2f} Amps</li>
            <li>Selected Panel Size: {amperage} Amps</li>
        </ul>
        
        <h4>Determination</h4>
        <p style="font-weight: bold; color: {'green' if is_adequate else 'red'};">
            The selected {amperage}A panel is {'ADEQUATE' if is_adequate else 'NOT ADEQUATE'} for the calculated load.
            {'' if is_adequate else f' Consider upgrading to a {max(int((total_amps * 1.25 // 25) + 1) * 25, 100)}A panel.'}
        </p>
        """
        
        return result
    
    def determine_code_requirements(self, panel_data: Dict[str, Any]) -> str:
        """
        Determine relevant electrical code requirements.
        
        Args:
            panel_data: Panel specifications data
            
        Returns:
            HTML formatted code requirements
        """
        # Extract relevant data
        panel_type = panel_data.get("panel_type", "Main Service Panel")
        installation_type = panel_data.get("installation_type", "Upgrade Existing Panel")
        location = panel_data.get("location", "")
        height_requirement = panel_data.get("height_requirement", False)
        clearance_requirement = panel_data.get("clearance_requirement", False)
        dedicated_lighting = panel_data.get("dedicated_lighting", False)
        grounding = panel_data.get("grounding", "")
        
        # Generate code requirements based on data
        code_html = "<h3>NEC Code Requirements</h3>"
 
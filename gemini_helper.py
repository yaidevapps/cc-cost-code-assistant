import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

class GeminiInspector:
    def __init__(self, api_key=None):
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize the model with Gemini 2.0
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Set generation config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
    def prepare_image(self, image):
        """Prepare the image for Gemini API"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        max_size = 4096
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def analyze_image(self, image, chat):
        """Analyze an invoice image using existing chat session"""
        try:
            processed_image = self.prepare_image(image)
            
            prompt = """# Construction Invoice and Estimate Cost Code Analysis System

## System Context and Role
You are a specialized construction accounting assistant with expertise in residential custom home building. Your primary function is to analyze invoice and estimate images and assign appropriate cost codes based on the standardized cost code structure for residential construction projects. You have been trained on an extensive database of construction terminology, common materials, and standard building practices.

## Core Responsibilities
1. Extract relevant information from invoice and estimate images
2. Analyze line items and assign cost codes
3. Generate organized tabular outputs
4. Explain reasoning for cost code assignments
5. Flag any uncertain classifications or poor text extractions for review

## Cost Code Structure
The following structure should be used exclusively for all classifications:
- 20100-Right of Way
- 30200-Permit & Plan Review
- 30250-Water Fees
- 30300-Electrical Fees
- 30350-City Sewer Fees
- 30400-Gas Fees
- 30500-Temporary Power
- 31000-Engineering Architectural
- 31100-Special Inspections
- 31500-Plan Copies
- 40000-Survey Stake Lot
- 40100-Clear Lot
- 40150-Temp Fencing
- 40200-Erosion Control
- 40225-Building Dryout
- 40250-Floor Protection
- 40300-Demo
- 40303-Asbestos Test
- 40304-Lead Test
- 40305-Abatement
- 40310-OnSite Storage
- 40325-Material Pickup
- 40350-Debris Removal
- 40375-Dust Control
- 40380-Traffic Control
- 40400-Excavate
- 40410-Oil Tank Removal
- 40420-Piles
- 40450-Rockeries
- 40500-Post Tension Concrete
- 40555-Concrete Cut
- 40600-Foundation Hardware (vents-wells-well covers-etc)
- 40650-Foundation Labor
- 40950-Foundation Concrete
- 41050-Foundation Pump
- 41600-Foundation Waterproofing
- 41800-Drainage
- 41900-Utilities Install
- 42000-Sewer Septic System
- 42100-Backfill
- 42200-Export
- 42300-Import
- 42400-Final Grade
- 50000-Lumber
- 50020-Trusses
- 50040-Structural Steel
- 50045-Crane Service
- 50050-Framing Labor 90%
- 50060-Framing Labor 10% Retention
- 50750-Framing Hardware
- 51400-Scaffolding
- 51750-Fireplaces
- 51800-GreenRoof
- 51850-Roofing
- 51900-Solar
- 51950-Fireplace Face Veneer
- 51970-Architectural Steel
- 52000-Interior Flatwork Material
- 52100-Interior Flatwork Labor
- 52150-Interior Flatwork Pump
- 52175-Garage Floors
- 52200-Gutters/Downspouts
- 52300-Windows
- 52310-Skylights
- 52340-Window Installation
- 52345-Window Screens
- 52350-Window Coverings
- 52400-Door Exterior Multislide Folding
- 52405-Door Exterior Multislide Folding Installation
- 52410-Door Pans
- 52420-Doors Exterior
- 52430-Entry Door
- 52450-Dock
- 52500-Elevator
- 52600-Deck Material
- 52610-Deck Labor
- 52650-Deck Waterproofing
- 52655-Deck Pavers
- 52660-Precast Concrete Treads Material
- 52670-Precast Concrete Treads Labor
- 52700-Pool
- 52800-Siding
- 52900-Exterior Metal Railing
- 53000-Masonry
- 54000-Awnings
- 60000-Plumbing Rough
- 60200-Electrical Rough
- 60220-Low Voltage RoughIn
- 60230-Generator
- 60250-Vacuum System
- 60300-Gas Piping
- 60350-Gas Heaters
- 60400-HVAC RoughIn
- 60500-Fire Sprinkler
- 60900-Insulation
- 70010-Plane and Butt Strip
- 70050-Drywall
- 70200-Automated Gates
- 70400-Garage Doors
- 70450-Garage Floor Coating
- 70500-Paint Exterior
- 70600-Paint Interior
- 70615-Wallpaper Material
- 70620-Wallpaper Labor
- 70625-Cabinets
- 70650-Interior Doors Millwork
- 70675-Mantel Material Only
- 70700-Interior Railing
- 70750-Custom Staircase
- 70810-Hardwood
- 70850-Finish Carpentry
- 71100-Paint Millwork
- 71200-Vinyl Flooring
- 71250-Polished Concrete
- 71375-Tile Material & Install
- 71400-PLam
- 71600-Marble/Granite/QuartzSlabs
- 71900-Light Fixtures
- 72000-Appliances
- 72050-Appliances Installation
- 72100-Finish Hardware Deliver
- 72200-Weatherstrip
- 72400-Mirrors/ShowerDoor
- 72450-Specialty Mirrors
- 72500-Insulation Blow
- 72550-Blower Door Test
- 72600-Electrical Trim
- 72620-Low Voltage Trim
- 72700-Plumbing Trim
- 72750-Plumbing Fixtures
- 72755-Hot Tub
- 73175-HVAC Duct Cleaning
- 73180-HVAC Trim
- 73200-Carpet
- 73300-Closet Shelving
- 73350-Wine Cellar/Closet
- 74000-Alarm System
- 80000-Exterior Flatwork Labor
- 80100-Exterior Flatwork Material
- 80200-Exterior Flatwork Pump
- 80250-Asphalt
- 80300-Pavers Landscaping
- 80400-Irrigation
- 80500-Landscape Lighting
- 80600-Landscaping
- 80700-Fencing
- 81000-Gutter Cleaning
- 85000-Drywall TouchUp
- 85100-Paint TouchUp
- 85200-Final Clean
- 86100-Bike Racks
- 86150-Mail Box Stands
- 90005-Carpentry/Pickup Framing Inhouse
- 91000-Misc Labor
- 91010-Floor Prep
- 91015-Final Detail
- 91200-ReCleans
- 91300-Landscape Maintenance
- 91400-Utility Usage
- 91600-Street Sweeping
- 91700-Sanican Rental
- 91800-Rental Tools Misc
- 91801-Job Shack
- 91802-Development Impact Fees
- 91850-Misc Materials

## Required Image Analysis Tasks
1. Extract:
   - Vendor name
   - Invoice or estimate date
   - Line item descriptions and amounts
   - Any additional context or notes on the invoice or estimate

## Output Format
Generate results with two primary components:

### [Invoice or Estimate] Details
| Field | Value |
|-------|-------|
| Vendor | [Vendor Name or "Not Detected"] |
| Vendor Address | [Vendor Address or "Not Detected"] |
| Vendor Phone | [Vendor Phone or "Not Detected"] |
| Vendor Email | [Vendor Email or "Not Detected"] |
| [Invoice or Estimate] Number | [Number or "Not Detected"] |
| [Invoice or Estimate] Date | [Date or "Not Detected"] |
| Due Date | [Date or "Not Detected"] |
| [Invoice or Estimate] Total | [Amount or "Not Detected"] |
| Project | [Project Name or "Not Detected"] |


### Line Item Classification
Generate a markdown table with the following columns:
| Line Item | Amount | Assigned Cost Code | Code Description | Confidence | Reasoning |


## Classification Guidelines
1. Always reference the provided cost code structure exactly as given
2. When multiple codes could apply:
   - Consider any provided project phase context
   - Use the more specific code when available
   - Document reasoning for the chosen classification
3. For items that could span multiple categories:
   - Break down into sub-components if possible
   - Assign to primary purpose if breakdown not feasible

## Confidence Levels and Uncertainty Handling
Express confidence in classifications as:
- High (>90%): Clear match to cost code description
- Medium (70-90%): Reasonable match with some assumptions
- Low (<70%): Significant uncertainty in classification

When confidence is Low:
1. Flag for manual review
2. Provide multiple possible code options
3. Explain specific sources of uncertainty or extraction issues

## Anti-hallucination Protocols
1. Only use cost codes explicitly present in the provided structure
2. Never invent or assume the existence of additional codes
3. When uncertain, explicitly state assumptions and limitations
4. If an item cannot be confidently classified or extracted, mark it for review rather than making an uncertain assignment

## Response Structure
1. Acknowledge the invoice or estimate image has been analyzed
2. Present the extracted invoice or estimate details
3. Display classified results in the specified table format
4. Explain any notable classification decisions or extraction issues
5. List any items flagged for review
6. Suggest improvements for future image submissions
7. Invite clarifying questions about specific classifications or extractions

Please analyze this invoice or estimate image:
"""
            
            # Send the message with image to the existing chat
            response = chat.send_message([prompt, processed_image])
            return response.text
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}\nDetails: Please ensure your API key is valid and you're using a supported image format."

    def start_chat(self):
        """Start a new chat session"""
        try:
            return self.model.start_chat(history=[])
        except Exception as e:
            return None

    def send_message(self, chat, message):
        """Send a message to the chat session"""
        try:
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            return f"Error sending message: {str(e)}"
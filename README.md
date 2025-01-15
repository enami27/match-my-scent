# MATCH MY SCENT

A ML-powered perfume recommendation system that uses computer vision to match images with fragrance profiles. By analyzing visual elements and comparing them with detailed perfume characteristics, providing personalized fragrance recommendations.

## Features

- Image-based perfume recommendations using Open AI's CLIP model
- Detailed mood categorization system for fragrances based on notes
- Comprehensive perfume analysis including top, middle and base notes, sillage, and longevity
- Customizable filtering options (gender, sillage, longevity)
- Confidence scoring system for recommendations
- Advanced logging system for debugging and monitoring

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pillow (PIL)
- Pandas
- NumPy
- scikit-learn
- CUDA-compatible GPU (optional but faster)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch transformers pillow pandas numpy scikit-learn
```

## Data Format

Your perfume data should be in an Excel file (`perfumes.xlsx`) with the following columns:
- perfume_brand
- perfume_name
- top_notes
- middle_notes
- base_notes
- perfume_gender
- perfume_sillage
- perfume_longevity

You can use the provided dataset or replace it with yours, just make sure it has the same columns (very important)

## Usage

Basic usage example:

```python
from perfume_matcher import PerfumeMatcher

# Initialize the matcher
matcher = PerfumeMatcher('perfumes.xlsx')

# Get recommendations
recommendations = matcher.get_recommendation(
    image_path="your_image.jpg",
    top_n=5,
    gender_filter="unisex"  # Optional
)

# Print recommendations
for rec in recommendations:
    print(f"Brand: {rec.brand}")
    print(f"Name: {rec.name}")
    print(f"Confidence Score: {rec.confidence_score}%")
```

## How It Works

1. **Image Processing**: The system uses OpenAI's CLIP model to extract visual features from your image

2. **Perfume Analysis**: Each perfume in the database is analyzed based on:
   - Note composition (top, middle, base notes)
   - Mood categories (e.g., fresh_sporty, elegant_sophisticated)
   - Performance characteristics (sillage, longevity)

3. **Matching Process**: The system:
   - Compares image features with perfume descriptions
   - Calculates similarity scores
   - Applies any specified filters
   - Generates confidence scores based on data completeness and similarity

## Mood Categories

The system includes 14 sophisticated mood categories, each with specific note associations:
- Fresh/Sporty
- Elegant/Sophisticated
- Romantic/Intimate
- Mysterious/Dark
- Tropical/Exotic
- And more...

## Custom Configuration

You can customize the recommendation process with filters:
```python
recommendations = matcher.get_recommendation(
    image_path="your_image.jpg",
    top_n=3,
    gender_filter="feminine",
    sillage_filter="moderate",
    longevity_filter="long-lasting"
)
```

## Output 

Each recommendation includes:
- Brand and name
- Similarity score (0-100%)
- Confidence score (0-100%)
- Mood category
- Complete note breakdown
- Gender classification
- Sillage and longevity ratings

## Debugging

The system includes comprehensive logging:
- All major operations are logged
- Error handling with detailed messages
- Performance monitoring
- Process tracking

## Troubleshooting

Common issues and solutions:
1. **Image not found**: Ensure the image path is correct and accessible
2. **No recommendations**: Check if your filters aren't too restrictive
3. **CUDA errors**: Ensure your GPU drivers are up to date or use CPU mode
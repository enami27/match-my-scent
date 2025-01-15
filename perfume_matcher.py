import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
import logging
import os
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerfumeRecommendation:
    brand: str
    name: str
    similarity: float
    mood: str
    top_notes: str
    middle_notes: str
    base_notes: str
    gender: str
    sillage: str
    longevity: str
    confidence_score: float

# Perfume matcher class
class PerfumeMatcher:
    def __init__(self, perfume_data_path: str, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.scaler = MinMaxScaler()
        
        # Detailed mood categories with weighted note associations for more acdurate mood assignment
        self.mood_categories = {
            'fresh_sporty': {
                'description': "A vibrant, energetic scent with citrus and green notes, perfect for active lifestyles.",
                'note_weights': {
                    'top': ['citrus', 'bergamot', 'lemon', 'lime', 'green'],
                    'middle': ['marine', 'ozonic', 'herbal'],
                    'base': ['musk', 'woody']
                }
            },
            'elegant_sophisticated': {
                'description': "A refined, luxurious fragrance with floral and woody notes.",
                'note_weights': {
                    'top': ['aldehydes', 'bergamot', 'neroli'],
                    'middle': ['rose', 'jasmine', 'iris', 'ylang'],
                    'base': ['sandalwood', 'vetiver', 'amber']
                }
            },
            'romantic_intimate': {
                'description': "A warm, sensual fragrance with sweet and oriental notes.",
                'note_weights': {
                    'top': ['vanilla', 'tonka bean', 'amber'],
                    'middle': ['rose', 'jasmine', 'orange blossom'],
                    'base': ['musk', 'amber', 'vanilla']
                }
            },

            'mysterious_dark': {
                'description': "A deep, enigmatic fragrance with smoky and spicy accords.",
                'note_weights': {
                    'top': ['incense', 'black pepper', 'spices', 'smoke'],
                    'middle': ['leather', 'tobacco', 'oud', 'dark woods'],
                    'base': ['patchouli', 'amber', 'dark musk', 'vanilla']
                }
            },

            'tropical_exotic': {
                'description': "A vibrant, tropical fragrance with exotic fruits and flowers.",
                'note_weights': {
                    'top': ['mango', 'pineapple', 'coconut', 'passion fruit'],
                    'middle': ['frangipani', 'tiare', 'ylang ylang', 'orchid'],
                    'base': ['vanilla', 'white musk', 'tropical woods']
                }
            },

            'aromatic_herbal': {
                'description': "A fresh, herbal fragrance with aromatic plants and spices.",
                'note_weights': {
                    'top': ['lavender', 'rosemary', 'sage', 'mint'],
                    'middle': ['thyme', 'basil', 'artemisia', 'fennel'],
                    'base': ['oakmoss', 'cedar', 'vetiver']
                }
            },

            'gourmand_sweet': {
                'description': "A delicious, sweet fragrance with edible and dessert-like notes.",
                'note_weights': {
                    'top': ['caramel', 'chocolate', 'coffee', 'almond'],
                    'middle': ['vanilla', 'tonka bean', 'praline', 'coconut'],
                    'base': ['benzoin', 'amber', 'musk', 'vanilla']
                }
            },

            'woody_earthy': {
                'description': "A grounding, natural fragrance with wood and earth elements.",
                'note_weights': {
                    'top': ['cedar', 'pine', 'cypress'],
                    'middle': ['sandalwood', 'patchouli', 'vetiver'],
                    'base': ['oakmoss', 'amber', 'musk']
                }
            },

            'powdery_soft': {
                'description': "A delicate, soft fragrance with powdery and cosmetic notes.",
                'note_weights': {
                    'top': ['violet', 'iris', 'heliotrope'],
                    'middle': ['rose', 'powder', 'makeup notes'],
                    'base': ['vanilla', 'musk', 'benzoin']
                }
            },

            'aquatic_marine': {
                'description': "A fresh, watery fragrance with marine and oceanic notes.",
                'note_weights': {
                    'top': ['sea notes', 'marine', 'salt', 'aquatic'],
                    'middle': ['seaweed', 'lotus', 'water lily'],
                    'base': ['musk', 'driftwood', 'ambergris']
                }
            },

            'oriental_spicy': {
                'description': "A rich, opulent fragrance with oriental spices and resins.",
                'note_weights': {
                    'top': ['cardamom', 'saffron', 'cinnamon'],
                    'middle': ['oud', 'rose', 'incense'],
                    'base': ['amber', 'vanilla', 'labdanum']
                }
            },

            'citrus_bright': {
                'description': "A zesty, energetic fragrance with citrus fruits and bright notes.",
                'note_weights': {
                    'top': ['bergamot', 'lemon', 'orange', 'grapefruit'],
                    'middle': ['neroli', 'petitgrain', 'verbena'],
                    'base': ['white musk', 'light woods']
                }
            },

            'leather_suede': {
                'description': "A sophisticated, animalic fragrance with leather and suede notes.",
                'note_weights': {
                    'top': ['leather', 'suede', 'saffron'],
                    'middle': ['tobacco', 'iris', 'styrax'],
                    'base': ['amber', 'vanilla', 'musk']
                }
            },
            'fresh_clean': {
                'description': "A crisp, clean fragrance with light and airy notes.",
                'note_weights': {
                    'top': ['lavender', 'citrus', 'marine'],
                    'middle': ['cotton', 'white flowers', 'powdery'],
                    'base': ['white musk', 'cedar', 'soap']
                }
            }
        }


        self.perfume_df = self._preprocess_perfume_data(perfume_data_path)
        self._initialize_embeddings()
        logger.info("PerfumeMatcher initialization completed")
    # preprocess perfume data with mood category and rich description
    def _preprocess_perfume_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Perfume data file not found: {file_path}")
        
        logger.info(f"Loading perfume data from {file_path}")
        df = pd.read_excel(file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Standardize note formatting
        for notes_col in ['top_notes', 'middle_notes', 'base_notes']:
            df[notes_col] = df[notes_col].apply(self._standardize_notes)
        
        # Assign moods based on note composition
        df['mood_category'] = df.apply(self._assign_mood_category_weighted, axis=1)
        
        # Create rich descriptions for embedding
        df['description'] = df.apply(self._create_rich_description, axis=1)
        
        logger.info(f"Preprocessed {len(df)} perfume entries")
        return df

    # Standardize and clean up pnotes
    def _standardize_notes(self, notes: str) -> str:
        if pd.isna(notes) or not notes:
            return ""
        notes = notes.lower().strip()
        notes = notes.replace('notes', '').replace('note', '')
        return notes
    # Assign mood category using a weighted scoring system based on notes
    def _assign_mood_category_weighted(self, row: pd.Series) -> str:
        scores = {mood: 0 for mood in self.mood_categories.keys()}
        
        for mood, details in self.mood_categories.items():
            weights = details['note_weights']
            
            # Check top notes (highest weight)
            for note in weights['top']:
                if note in row['top_notes'].lower():
                    scores[mood] += 3
                    
            # Check middle notes (medium weight)
            for note in weights['middle']:
                if note in row['middle_notes'].lower():
                    scores[mood] += 2
                    
            # Check base notes (lower weight)
            for note in weights['base']:
                if note in row['base_notes'].lower():
                    scores[mood] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'fresh_clean'
    # creatd detailed description for embedding
    def _create_rich_description(self, row: pd.Series) -> str:
        return (
            f"{row['perfume_brand']} {row['perfume_name']} fragrance with "
            f"{row['top_notes']} in the top notes, "
            f"{row['middle_notes']} in the heart, and "
            f"{row['base_notes']} in the base. "
            f"A {row['mood_category'].replace('_', ' ')} scent with "
            f"{row['perfume_sillage']} sillage and {row['perfume_longevity']} longevity. "
            f"Designed for {row['perfume_gender']}."
        )

    # Initialize and cache embeddings
    def _initialize_embeddings(self):
        logger.info("Initializing perfume embeddings")
        with torch.no_grad():
            descriptions = self.perfume_df['description'].tolist()
            inputs = self.processor(
                text=descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            self.perfume_embeddings = self.model.get_text_features(**inputs)
            self.perfume_embeddings = self.perfume_embeddings / self.perfume_embeddings.norm(dim=-1, keepdim=True)
        logger.info("Embeddings initialization completed")

    # Preprocess image and get features
    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.squeeze()
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    # Get perfume recommendations with filtering and confidence score
    def get_recommendation(
        self,
        image_path: str,
        top_n: int = 3,
        gender_filter: Optional[str] = None,
        sillage_filter: Optional[str] = None,
        longevity_filter: Optional[str] = None
    ) -> List[PerfumeRecommendation]:

        try:
            logger.info(f"Processing recommendations for image: {image_path}")
            image_features = self.process_image(image_path)
            if image_features is None:
                return []

            # Calculate similarities
            with torch.no_grad():
                similarities = torch.matmul(self.perfume_embeddings, image_features.t()).squeeze()

            # Apply filters
            mask = torch.ones(len(similarities), dtype=torch.bool)
            if gender_filter:
                mask &= self.perfume_df['perfume_gender'].str.lower() == gender_filter.lower()
            if sillage_filter:
                mask &= self.perfume_df['perfume_sillage'].str.lower() == sillage_filter.lower()
            if longevity_filter:
                mask &= self.perfume_df['perfume_longevity'].str.lower() == longevity_filter.lower()

            filtered_similarities = similarities[mask]
            filtered_indices = torch.arange(len(similarities))[mask]

            if len(filtered_similarities) == 0:
                logger.warning("No perfumes found matching the filters")
                return []

            # Get top recommendations
            top_values, top_indices = torch.topk(filtered_similarities, min(top_n, len(filtered_similarities)))
            selected_indices = filtered_indices[top_indices].tolist()

            recommendations = []
            for idx, similarity_score in zip(selected_indices, top_values):
                perfume = self.perfume_df.iloc[idx]
                
                # Calculate confidence score
                data_completeness = sum(
                    1 for field in [
                        perfume['top_notes'],
                        perfume['middle_notes'],
                        perfume['base_notes']
                    ] if field and len(field) > 0
                ) / 3.0
                
                confidence_score = (similarity_score.item() * 0.7 + data_completeness * 0.3) * 100

                recommendations.append(PerfumeRecommendation(
                    brand=perfume['perfume_brand'],
                    name=perfume['perfume_name'],
                    similarity=float(similarity_score.cpu().numpy()) * 100,
                    mood=perfume['mood_category'],
                    top_notes=perfume['top_notes'],
                    middle_notes=perfume['middle_notes'],
                    base_notes=perfume['base_notes'],
                    gender=perfume['perfume_gender'],
                    sillage=perfume['perfume_sillage'],
                    longevity=perfume['perfume_longevity'],
                    confidence_score=round(confidence_score, 2)
                ))

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error in get_recommendation: {e}")
            return []

def main():
    try:
        # Initialize the matcher
        matcher = PerfumeMatcher('perfumes.xlsx')
        
        # Get recommendations
        recommendations = matcher.get_recommendation(
            "test_image.jpg",
            top_n=5,
            gender_filter=None  # Optional filters
        )
        
        # Print recommendations
        if recommendations:
            print("\nTop Perfume Recommendations:")
            print("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                print(f"\nRecommendation {i}:")
                print(f"Brand: {rec.brand}")
                print(f"Name: {rec.name}")
                print(f"Similarity Score: {rec.similarity:.2f}%")
                print(f"Confidence Score: {rec.confidence_score:.2f}%")
                print(f"Mood: {rec.mood.replace('_', ' ').title()}")
                print(f"Notes:")
                print(f"  Top: {rec.top_notes}")
                print(f"  Middle: {rec.middle_notes}")
                print(f"  Base: {rec.base_notes}")
                print(f"Gender: {rec.gender}")
                print(f"Sillage: {rec.sillage}")
                print(f"Longevity: {rec.longevity}")
                print("-" * 50)
        else:
            print("No recommendations found.")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print("An error occurred while generating recommendations.")

if __name__ == "__main__":
    main()
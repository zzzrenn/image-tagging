import torch
import clip
from PIL import Image
from typing import List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneAnalyzer:
    def __init__(self):
        self.categories = {
            'weather': [
                'sunny', 'cloudy', 'rainy', 'snowy', 'foggy', 'clear sky', 'stormy', 'overcast', 'partly cloudy', 'misty'
            ],
            'altitude': ['5 meter', '50 meter', '100 meter'],
            'point_of_view': [
                'low-angle', "high-angle", "birds-eye-view"
            ],
            'location': [
                'indoor', 'beach', 'mountain', 'forest', 'desert', 'city street',
                'park', 'industrial', 'rural', 'highway'
            ],
            'time_of_day': [
                'day', 'night'
            ]
        }
        
        # Load CLIP model
        logger.info("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device, download_root="weights")
        logger.info(f"CLIP model loaded successfully. Using device: {self.device}")
        
    def _encode_text_descriptions(self, descriptions: List[str]) -> torch.Tensor:
        """Encode text descriptions using CLIP"""
        text = clip.tokenize(descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using CLIP"""
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _calculate_similarities(self, image_features: torch.Tensor, 
                              text_features: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarities between image and text features"""
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarities.squeeze()

    def _get_top_predictions(self, similarities: torch.Tensor, 
                           descriptions: List[str], top_k: int = 1) -> List[Dict]:
        """Get top k predictions with their probabilities"""
        values, indices = similarities.topk(top_k)
        return [
            {
                'label': descriptions[idx],
                'probability': float(val) * 100
            }
            for val, idx in zip(values.cpu(), indices.cpu())
        ]

    def analyze_image(self, image: Image.Image) -> Dict[str, List[Dict]]:
        """Analyze image across all categories"""
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Get image features
            logger.info("Encoding image...")
            image_features = self._encode_image(image)
            
            results = {}
            
            # Analyze each category
            for category_name, descriptions in tqdm(self.categories.items(), 
                                                  desc="Analyzing categories"):
                try:
                    # Encode text descriptions
                    text_features = self._encode_text_descriptions(descriptions)
                    
                    # Calculate similarities
                    similarities = self._calculate_similarities(image_features, 
                                                             text_features)
                    
                    # Get top predictions
                    results[category_name] = self._get_top_predictions(
                        similarities, descriptions
                    )
                    
                except Exception as e:
                    logger.error(f"Error analyzing category {category_name}: {str(e)}")
                    results[category_name] = []
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            raise

    def get_categories(self) -> Dict[str, List[str]]:
        """Return available categories and their descriptions"""
        return self.categories
    
if __name__ == "__main__":
    # test
    model = SceneAnalyzer()
    image = Image.open("../datasets/VisDrone2019-DET-val/images/0000333_02549_d_0000014.jpg")
    result = model.analyze_image(image)
    print(result)
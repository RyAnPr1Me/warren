#!/usr/bin/env python3
"""
Test the enhanced LSTM model on 10 years of TSLA data
"""

import sys
import os
sys.path.append('/Users/rmanzo28/Downloads/untitled folder')

from src.models.lstm_predictor import EnhancedLSTMPredictor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Test TSLA model with 10 years of data"""
    
    print("🚀 Testing Enhanced LSTM Model on 10 Years of TSLA Data")
    print("=" * 80)
    
    try:
        # Initialize the predictor
        logger.info("Initializing TSLA predictor...")
        predictor = EnhancedLSTMPredictor(symbol="TSLA")
        
        # Train on 10 years of data
        logger.info("Training model on 10 years of TSLA data...")
        print("\n📊 Starting training with all improvements:")
        print("   • MSE loss for R² optimization")
        print("   • Log returns with improved clipping") 
        print("   • Simplified 12-feature set")
        print("   • Automatic architecture selection")
        print("   • Comprehensive evaluation with percentages")
        print("\n" + "=" * 80)
        
        # Use 10y period for maximum data
        results = predictor.train_enhanced_model(period="10y")
        
        if 'error' in results:
            print(f"\n❌ Training failed: {results['error']}")
            return False
        
        print("\n🎉 Training completed successfully!")
        print(f"📈 Final Results Summary:")
        print(f"   • Validation R²: {results.get('val_r2', 'N/A'):.4f}")
        print(f"   • Test R²: {results.get('test_r2', 'N/A'):.4f}")
        print(f"   • Direction Accuracy: {results.get('test_direction_accuracy', 0)*100:.1f}%")
        print(f"   • Training Time: {results.get('epochs_trained', 'N/A')} epochs")
        print(f"   • Model Parameters: {results.get('total_parameters', 'N/A'):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ TSLA model test completed successfully!")
    else:
        print("\n❌ TSLA model test failed!")

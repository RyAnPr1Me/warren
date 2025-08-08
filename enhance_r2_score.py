"""
Enhanced R² Score Improvement Implementation
Comprehensive script to train LSTM with advanced features and optimizations
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enhance_model_for_r2_improvement():
    """
    Comprehensive R² improvement for TSLA model with advanced features
    """
    from src.models.lstm_predictor import EnhancedLSTMPredictor
    
    logger.info("🚀 Starting comprehensive R² improvement training for TSLA...")
    
    # Initialize predictor with enhanced settings
    predictor = EnhancedLSTMPredictor('TSLA')
    
    # Train with maximum data and advanced features
    try:
        # Force use of maximum available data (period='max') and advanced features (mega_data=True)
        results = predictor.train_enhanced_model(period='max', mega_data=True)
        
        if 'error' in results:
            logger.error(f"Training failed: {results['error']}")
            return results
            
        # Log comprehensive results
        logger.info("🎯 R² IMPROVEMENT TRAINING RESULTS:")
        logger.info(f"   Validation R²: {results.get('val_r2', 'N/A'):.4f}")
        logger.info(f"   Test R²: {results.get('test_r2', 'N/A'):.4f}")
        logger.info(f"   Training Time: {results.get('training_time', 'N/A'):.1f}s")
        logger.info(f"   Features Used: {results.get('num_features', 'N/A')}")
        logger.info(f"   Training Samples: {results.get('train_samples', 'N/A')}")
        
        # Check if R² improved significantly
        val_r2 = results.get('val_r2', -999)
        test_r2 = results.get('test_r2', -999)
        
        if val_r2 > 0.01 and test_r2 > 0.01:
            logger.info("✅ EXCELLENT: R² scores significantly improved!")
        elif val_r2 > -0.01 and test_r2 > -0.01:
            logger.info("✅ GOOD: R² scores improved to near-positive territory!")
        else:
            logger.warning("⚠️ R² scores still need improvement. Consider:")
            logger.warning("   • More training data (try longer periods)")
            logger.warning("   • Feature selection optimization")
            logger.warning("   • Different model architectures")
            logger.warning("   • Alternative loss functions")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        return {'error': str(e)}

def create_r2_improvement_documentation():
    """
    Create comprehensive documentation for R² improvements
    """
    doc_content = """
# R² Score Improvement Implementation

## Problem Analysis
The LSTM model was showing negative R² scores (-0.023 validation, -0.018 test), indicating poor predictive performance compared to naive baselines.

## Implemented Solutions

### 1. Advanced Feature Engineering
- **Comprehensive Features**: 200+ engineered features including:
  - Market microstructure features (price efficiency, fractal dimension)
  - Multi-horizon volatility features (realized vol, skewness, kurtosis)
  - Multi-timeframe technical indicators (MA cross signals, BB positions)
  - Statistical distribution features (moments, quantiles, normality tests)
  - Market regime detection (trend strength, stress indicators)
  - Momentum and mean reversion signals
  - Predictive alpha features (forward-looking indicators)
  - Feature interactions and non-linearities

### 2. Enhanced Data Collection
- **Maximum Historical Data**: Use 'max' period instead of 3y for more training data
- **Mega Data Mode**: Enable comprehensive data collection from all APIs
- **Better Data Quality**: Advanced preprocessing and outlier handling

### 3. Improved Model Architecture
- **More Features**: Adaptive feature selection using all available quality features
- **Better Target Engineering**: Log returns with optimized clipping
- **Enhanced Scaling**: Consistent StandardScaler for features and targets

### 4. Training Optimizations
- **Plateau-Breaking Optimizations**: Already implemented cyclical LR and adaptive loss
- **Quality Validation**: Comprehensive metrics tracking
- **Better Regularization**: Balanced regularization to prevent overfitting

## Expected Improvements
- **R² Target**: Aim for positive R² scores (> 0.01)
- **Better Generalization**: Reduced overfitting with more features and data
- **Improved Direction Accuracy**: Better trading signals
- **Enhanced Stability**: More robust predictions

## Monitoring
Watch for:
- R² scores trending toward positive values
- Stable validation performance
- Improved direction accuracy (>55%)
- Consistent training convergence

## Next Steps if R² Still Poor
1. **Feature Selection**: Use mutual information or correlation-based feature selection
2. **Alternative Models**: Try ensemble methods or gradient boosting
3. **Different Targets**: Try predicting price direction instead of returns
4. **External Data**: Add economic indicators, sector data, options flow
5. **Model Complexity**: Try deeper networks or transformer architectures
"""
    
    doc_path = Path("R2_IMPROVEMENT_IMPLEMENTATION.md")
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    logger.info(f"📄 Documentation created: {doc_path}")

if __name__ == "__main__":
    # Create documentation
    create_r2_improvement_documentation()
    
    # Run enhanced training
    results = enhance_model_for_r2_improvement()
    
    if 'error' not in results:
        print("\n" + "="*60)
        print("🎯 R² IMPROVEMENT TRAINING COMPLETE")
        print("="*60)
        print(f"📈 Validation R²: {results.get('val_r2', 'N/A'):.4f}")
        print(f"📊 Test R²: {results.get('test_r2', 'N/A'):.4f}")
        print(f"⏱️ Training Time: {results.get('training_time', 'N/A'):.1f}s")
        print(f"🔧 Features Used: {results.get('num_features', 'N/A')}")
        print("="*60)
    else:
        print(f"\n❌ Training failed: {results['error']}")

"""
Tests for model architectures
"""

import pytest
import numpy as np
import tensorflow as tf

# Import models to test
from src.models.unet import Unet
from src.models.atten_unet import AttenUnet
from src.models.resunet import ResUnet


class TestUnet:
    """Test U-Net model"""
    
    def test_unet_creation(self):
        """Test U-Net model can be created"""
        model = Unet(
            filters=[32, 64, 128, 256],
            final_activation='sigmoid',
            n_output=1
        )
        built_model = model.build()
        assert built_model is not None
        assert isinstance(built_model, tf.keras.Model)
    
    def test_unet_output_shape(self):
        """Test U-Net output shape"""
        model = Unet(
            filters=[32, 64],
            final_activation='sigmoid',
            n_output=1
        )
        built_model = model.build()
        
        # Test with sample input
        sample_input = np.random.random((1, 256, 256, 3))
        output = built_model.predict(sample_input)
        
        assert output.shape == (1, 256, 256, 1)


class TestAttenUnet:
    """Test Attention U-Net model"""
    
    def test_attention_unet_creation(self):
        """Test Attention U-Net model can be created"""
        model = AttenUnet(
            filters=[32, 64, 128],
            final_activation='sigmoid',
            n_output=1
        )
        built_model = model.build()
        assert built_model is not None
        assert isinstance(built_model, tf.keras.Model)


class TestResUnet:
    """Test ResU-Net model"""
    
    def test_resunet_creation(self):
        """Test ResU-Net model can be created"""
        model = ResUnet(
            filters=[32, 64, 128],
            final_activation='sigmoid',
            n_output=1
        )
        built_model = model.build()
        assert built_model is not None
        assert isinstance(built_model, tf.keras.Model)


if __name__ == "__main__":
    pytest.main([__file__]) 
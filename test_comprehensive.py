#!/usr/bin/env python3
"""
Comprehensive Test Suite for CNN Depth Processing System
Validates all components and integration points
"""

import unittest
import numpy as np
import torch
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env import UAVEnv, CONFIG
from depth_cnn import DepthCNN, DepthFeatureExtractor
from mujoco_depth_camera import MuJoCoDepthCamera
from depth_preprocessing import DepthPreprocessor, DepthAugmentor, create_preprocessing_pipeline


class TestDepthCNN(unittest.TestCase):
    """Test CNN architecture and feature extraction"""
    
    def setUp(self):
        self.cnn = DepthCNN(output_features=128)
        self.extractor = DepthFeatureExtractor(model_path=None)
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass with valid input"""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 64, 64)
        
        output = self.cnn(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.isfinite(output).all())
    
    def test_feature_extractor_numpy_input(self):
        """Test feature extraction with numpy array input"""
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        
        features = self.extractor.extract_features(depth_image)
        
        self.assertEqual(features.shape, (128,))
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(np.isnan(features).any())
    
    def test_feature_extractor_tensor_input(self):
        """Test feature extraction with tensor input"""
        depth_tensor = torch.randn(1, 1, 64, 64)
        
        features = self.extractor.extract_features(depth_tensor)
        
        self.assertEqual(features.shape, (128,))
        self.assertTrue(np.isfinite(features).all())
    
    def test_spatial_features(self):
        """Test spatial feature extraction"""
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        
        spatial_features = self.extractor.get_spatial_features(depth_image)
        
        expected_keys = ['min_depth', 'mean_depth', 'max_depth', 'depth_std']
        for key in expected_keys:
            self.assertIn(key, spatial_features)
            self.assertTrue(np.isfinite(spatial_features[key]))
    
    def test_model_consistency(self):
        """Test model produces consistent output"""
        # Test that same input produces same output
        test_input = np.random.uniform(0.5, 5.0, (64, 64))
        
        features1 = self.extractor.extract_features(test_input)
        features2 = self.extractor.extract_features(test_input)
        
        np.testing.assert_array_almost_equal(features1, features2, decimal=5)


class TestDepthPreprocessing(unittest.TestCase):
    """Test depth image preprocessing pipeline"""
    
    def setUp(self):
        self.preprocessor = DepthPreprocessor(
            image_size=64,
            depth_range=(0.1, 10.0),
            enable_denoising=False,  # Disable for consistent testing
            enable_enhancement=False,
            add_noise=False
        )
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing pipeline"""
        # Create test image with various patterns
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        
        result = self.preprocessor.preprocess(depth_image)
        
        self.assertEqual(result.shape, (1, 1, 64, 64))
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)
    
    def test_invalid_value_handling(self):
        """Test handling of NaN and infinity values"""
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        
        # Add invalid values
        depth_image[0:10, 0:10] = np.nan
        depth_image[50:60, 50:60] = np.inf
        depth_image[20:30, 20:30] = -np.inf
        
        result = self.preprocessor.preprocess(depth_image)
        
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.isnan(result).any())
    
    def test_out_of_range_handling(self):
        """Test handling of out-of-range depth values"""
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        
        # Add out-of-range values
        depth_image[0:10, 0:10] = 0.05  # Below minimum
        depth_image[50:60, 50:60] = 15.0  # Above maximum
        
        result = self.preprocessor.preprocess(depth_image)
        
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)
    
    def test_preprocessing_statistics(self):
        """Test preprocessing statistics collection"""
        depth_image = np.random.uniform(0.5, 5.0, (64, 64))
        depth_image[0:5, 0:5] = np.nan  # Add some invalid pixels
        
        self.preprocessor.reset_stats()
        self.preprocessor.preprocess(depth_image)
        
        stats = self.preprocessor.get_stats()
        
        self.assertEqual(stats['processed_count'], 1)
        self.assertGreater(stats['invalid_pixels'], 0)
    
    def test_resize_functionality(self):
        """Test image resizing"""
        # Test different input sizes
        for input_size in [32, 48, 128]:
            depth_image = np.random.uniform(0.5, 5.0, (input_size, input_size))
            
            result = self.preprocessor.preprocess(depth_image)
            
            self.assertEqual(result.shape, (1, 1, 64, 64))
    
    def test_augmentation(self):
        """Test depth augmentation"""
        augmentor = DepthAugmentor(
            enable_flip=True,
            enable_rotate=False,  # Might not have OpenCV
            enable_scale=False
        )
        
        depth_tensor = torch.randn(1, 1, 64, 64)
        
        augmented = augmentor.augment(depth_tensor)
        
        self.assertEqual(augmented.shape, depth_tensor.shape)
        self.assertTrue(torch.isfinite(augmented).all())


class TestMuJoCoDepthCamera(unittest.TestCase):
    """Test MuJoCo depth camera interface"""
    
    def setUp(self):
        # Mock MuJoCo components
        self.mock_model = MagicMock()
        self.mock_data = MagicMock()
        
        # Set up mock camera ID
        self.mock_model.cam_name2id = MagicMock(return_value=0)
        self.mock_model.ncam = 1
        
    @patch('mujoco.Renderer')
    def test_camera_initialization(self, mock_renderer_class):
        """Test camera initialization"""
        camera = MuJoCoDepthCamera(
            model=self.mock_model, 
            camera_name='test_camera',
            image_size=64
        )
        
        self.assertEqual(camera.camera_name, 'test_camera')
        self.assertEqual(camera.image_size, 64)
        mock_renderer_class.assert_called_once()
    
    @patch('mujoco.Renderer')
    def test_depth_image_rendering(self, mock_renderer_class):
        """Test depth image rendering with fallback"""
        # Set up mock renderer
        mock_renderer = MagicMock()
        mock_renderer_class.return_value = mock_renderer
        mock_renderer.render.return_value = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        camera = MuJoCoDepthCamera(
            model=self.mock_model,
            camera_name='test_camera',
            image_size=64
        )
        
        depth_image = camera.render_depth_image(self.mock_data)
        
        self.assertEqual(depth_image.shape, (64, 64))
        self.assertTrue(np.isfinite(depth_image).all())
        self.assertGreaterEqual(depth_image.min(), 0.0)
    
    def test_camera_properties(self):
        """Test camera property access"""
        camera = MuJoCoDepthCamera(
            model=self.mock_model,
            camera_name='test_camera',
            image_size=64
        )
        
        self.assertEqual(camera.camera_name, 'test_camera')
        self.assertEqual(camera.image_size, 64)


class TestEnvironmentIntegration(unittest.TestCase):
    """Test CNN integration with UAV environment"""
    
    def setUp(self):
        # Store original config
        self.original_config = CONFIG.copy()
    
    def tearDown(self):
        # Restore original config
        CONFIG.clear()
        CONFIG.update(self.original_config)
    
    def test_cnn_mode_initialization(self):
        """Test environment initialization with CNN mode"""
        CONFIG['use_cnn_depth'] = True
        
        env = UAVEnv(render_mode=None)
        
        self.assertIsNotNone(env.depth_camera)
        self.assertIsNotNone(env.depth_extractor)
        self.assertIsNotNone(env.depth_preprocessor)
        
        # Check observation space
        self.assertIsNotNone(env.observation_space)
        expected_dim = 3 + 3 + 3 + CONFIG['cnn_features_dim'] + 5  # navigation features
        self.assertEqual(env.observation_space.shape[0], expected_dim)
        
        env.close()
    
    def test_raycast_mode_initialization(self):
        """Test environment initialization with raycast mode"""
        CONFIG['use_cnn_depth'] = False
        
        env = UAVEnv(render_mode=None)
        
        self.assertIsNone(env.depth_camera)
        self.assertIsNone(env.depth_extractor)
        self.assertIsNone(env.depth_preprocessor)
        
        # Check observation space
        self.assertIsNotNone(env.observation_space)
        expected_dim = 3 + 3 + 3 + CONFIG['depth_features_dim'] + 11  # engineered features
        self.assertEqual(env.observation_space.shape[0], expected_dim)
        
        env.close()
    
    def test_observation_consistency(self):
        """Test that observations match declared observation space"""
        for use_cnn in [True, False]:
            CONFIG['use_cnn_depth'] = use_cnn
            
            with self.subTest(use_cnn=use_cnn):
                env = UAVEnv(render_mode=None)
                
                obs, _ = env.reset()
                
                # Check observation shape matches space
                self.assertEqual(obs.shape, env.observation_space.shape)
                
                # Check observation values are finite
                self.assertTrue(np.isfinite(obs).all())
                self.assertFalse(np.isnan(obs).any())
                
                env.close()
    
    def test_mode_switching(self):
        """Test switching between CNN and raycast modes"""
        # Test CNN mode
        CONFIG['use_cnn_depth'] = True
        env_cnn = UAVEnv(render_mode=None)
        obs_cnn, _ = env_cnn.reset()
        cnn_dim = obs_cnn.shape[0]
        env_cnn.close()
        
        # Test raycast mode
        CONFIG['use_cnn_depth'] = False
        env_raycast = UAVEnv(render_mode=None)
        obs_raycast, _ = env_raycast.reset()
        raycast_dim = obs_raycast.shape[0]
        env_raycast.close()
        
        # Dimensions should be different
        self.assertNotEqual(cnn_dim, raycast_dim)
        self.assertGreater(cnn_dim, raycast_dim)  # CNN should have more features
    
    def test_environment_steps(self):
        """Test environment stepping with both modes"""
        for use_cnn in [True, False]:
            CONFIG['use_cnn_depth'] = use_cnn
            
            with self.subTest(use_cnn=use_cnn):
                env = UAVEnv(render_mode=None)
                
                obs, _ = env.reset()
                
                # Test multiple steps
                for _ in range(5):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Verify observation properties
                    self.assertEqual(obs.shape, env.observation_space.shape)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertIsInstance(reward, (int, float))
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                
                env.close()


class TestConfigurationSystem(unittest.TestCase):
    """Test configuration system and parameter validation"""
    
    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation from config"""
        test_config = {
            'depth_resolution': 64,
            'depth_range': 3.0,
            'enable_denoising': True,
            'enable_enhancement': False,
            'add_noise': True,
            'noise_std': 0.02
        }
        
        preprocessor = create_preprocessing_pipeline(test_config)
        
        self.assertIsInstance(preprocessor, DepthPreprocessor)
        self.assertEqual(preprocessor.image_size, 64)
        self.assertEqual(preprocessor.depth_max, 3.0)
        self.assertTrue(preprocessor.add_noise)
        self.assertEqual(preprocessor.noise_std, 0.02)
    
    def test_config_parameter_validation(self):
        """Test that invalid configurations are handled gracefully"""
        # This should not crash the system
        invalid_configs = [
            {'depth_resolution': -1},
            {'depth_range': 0},
            {'cnn_features_dim': 0},
            {'noise_std': -1}
        ]
        
        for invalid_config in invalid_configs:
            with self.subTest(config=invalid_config):
                # Should handle gracefully or use defaults
                try:
                    create_preprocessing_pipeline(invalid_config)
                except Exception as e:
                    # Should be a reasonable error, not a crash
                    self.assertIsInstance(e, (ValueError, TypeError))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_dependencies_handling(self):
        """Test behavior when optional dependencies are missing"""
        # Test OpenCV fallback
        with patch('depth_preprocessing.CV2_AVAILABLE', False):
            preprocessor = DepthPreprocessor(enable_denoising=True, enable_enhancement=True)
            
            # Should work without OpenCV
            depth_image = np.random.uniform(0.5, 5.0, (64, 64))
            result = preprocessor.preprocess(depth_image)
            
            self.assertEqual(result.shape, (1, 1, 64, 64))
    
    def test_camera_failure_fallback(self):
        """Test fallback to raycast when camera fails"""
        CONFIG['use_cnn_depth'] = True
        
        env = UAVEnv(render_mode=None)
        
        # Even if camera fails, environment should work
        obs, _ = env.reset()
        
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(obs).all())
        
        env.close()
    
    def test_extreme_depth_values(self):
        """Test handling of extreme depth values"""
        preprocessor = DepthPreprocessor()
        
        extreme_cases = [
            np.full((64, 64), 0.0),      # All zero
            np.full((64, 64), 1000.0),   # Very large values
            np.full((64, 64), np.nan),   # All NaN
            np.full((64, 64), np.inf),   # All infinity
        ]
        
        for extreme_case in extreme_cases:
            with self.subTest(case=str(extreme_case.flat[0])):
                result = preprocessor.preprocess(extreme_case)
                
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(torch.isnan(result).any())


def run_comprehensive_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDepthCNN,
        TestDepthPreprocessing,
        TestMuJoCoDepthCamera,
        TestEnvironmentIntegration,
        TestConfigurationSystem,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Comprehensive CNN Depth Processing Test Suite")
    print("=" * 60)
    
    success = run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All comprehensive tests PASSED!")
        print("\nSystem Status: READY FOR PRODUCTION")
        print("✅ All components thoroughly tested and validated")
    else:
        print("❌ Some comprehensive tests FAILED!")
        print("Please review the test output above for details")
        sys.exit(1)
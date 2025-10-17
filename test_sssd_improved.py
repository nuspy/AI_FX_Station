"""
Test script for ImprovedSSSDModel with diffusers schedulers.

Usage:
    python test_sssd_improved.py
"""
import torch
from src.forex_diffusion.models.sssd_improved import ImprovedSSSDModel
from src.forex_diffusion.config.sssd_config import SSSDConfig

def test_improved_sssd():
    """Test all schedulers with dummy data"""
    
    print("=" * 60)
    print("Testing ImprovedSSSDModel with Diffusers Schedulers")
    print("=" * 60)
    
    # Create minimal config
    config = SSSDConfig()
    
    # Test all schedulers
    schedulers = ["dpmpp", "ddim", "euler", "kdpm2"]
    
    for scheduler_name in schedulers:
        print(f"\nTesting {scheduler_name.upper()} scheduler...")
        
        try:
            # Create model
            model = ImprovedSSSDModel(config, scheduler_type=scheduler_name)
            model.eval()
            
            # Create dummy features (batch=2, seq=100, features=config.feature_dim)
            feature_dim = config.model.encoder.feature_dim
            dummy_features = {
                "5m": torch.randn(2, 100, feature_dim),
                "15m": torch.randn(2, 100, feature_dim),
                "1h": torch.randn(2, 100, feature_dim),
            }
            
            # Test inference with few samples (fast test)
            print(f"  Running inference with {scheduler_name}...")
            with torch.no_grad():
                predictions = model.inference_forward(
                    features=dummy_features,
                    horizons=[0, 1],  # 2 horizons
                    num_samples=10,   # Few samples for speed
                    num_steps=10      # Few steps for speed
                )
            
            # Validate output
            assert len(predictions) == 2, f"Expected 2 horizons, got {len(predictions)}"
            
            for h_idx, pred in predictions.items():
                assert "mean" in pred, "Missing 'mean' key"
                assert "std" in pred, "Missing 'std' key"
                assert "q05" in pred, "Missing 'q05' key"
                assert "q50" in pred, "Missing 'q50' key"
                assert "q95" in pred, "Missing 'q95' key"
                
                # Check shapes
                assert pred["mean"].shape == (2,), f"Wrong shape: {pred['mean'].shape}"
                
                # Check values are reasonable (not NaN, Inf)
                assert torch.isfinite(pred["mean"]).all(), "NaN/Inf in predictions"
            
            print(f"  [OK] {scheduler_name.upper()}: SUCCESS")
            print(f"     Mean prediction: {predictions[0]['mean'][0].item():.6f}")
            print(f"     Uncertainty (std): {predictions[0]['std'][0].item():.6f}")
            
        except Exception as e:
            print(f"  [FAIL] {scheduler_name.upper()}: FAILED - {e}")
            raise
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load existing SSSD checkpoint")
    print("2. Run benchmark_improvement() vs original")
    print("3. Compare inference speed (should be 5-10x faster)")
    print("4. Validate prediction quality on real data")


def test_compare_schedulers():
    """Test compare_schedulers method"""
    
    print("\n" + "=" * 60)
    print("Testing Scheduler Comparison")
    print("=" * 60)
    
    # Create model
    config = SSSDConfig()
    model = ImprovedSSSDModel(config, scheduler_type="dpmpp")
    model.eval()
    
    # Dummy features (use config feature_dim)
    feature_dim = config.model.encoder.feature_dim
    dummy_features = {
        "5m": torch.randn(1, 100, feature_dim),
        "15m": torch.randn(1, 100, feature_dim),
        "1h": torch.randn(1, 100, feature_dim),
    }
    
    # Compare all schedulers
    print("\nBenchmarking all schedulers (10 samples, 10 steps)...")
    results = model.compare_schedulers(
        features=dummy_features,
        horizons=[0],
        num_samples=10,
        num_steps=10
    )
    
    # Print results
    print("\nResults:")
    for name, data in sorted(results.items(), key=lambda x: x[1]['time_ms']):
        print(f"  {name.upper():8s}: {data['time_ms']:6.1f}ms")
    
    fastest = min(results.items(), key=lambda x: x[1]['time_ms'])
    print(f"\n[WINNER] Fastest: {fastest[0].upper()} ({fastest[1]['time_ms']:.1f}ms)")


if __name__ == "__main__":
    # Test basic functionality
    test_improved_sssd()
    
    # Test scheduler comparison
    test_compare_schedulers()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] SSSD DIFFUSERS INTEGRATION WORKING!")
    print("=" * 60)

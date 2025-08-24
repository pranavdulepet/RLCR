#!/usr/bin/env python3
"""
Test script to verify all required downloads work before running training
"""

import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_download():
    """Test if we can download the base model"""
    print("Testing model download...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
        print("✓ Tokenizer download successful")
        
        # Test model download (just config, not full model)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
        print("✓ Model config download successful")
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False
    return True

def test_dataset_download():
    """Test if we can download the datasets"""
    print("Testing dataset downloads...")
    try:
        # HotpotQA dataset
        dataset = load_dataset("mehuldamani/hotpot_qa", split="train[:10]")
        print(f"✓ HotpotQA dataset download successful ({len(dataset)} samples)")
        
        # Math dataset  
        dataset = load_dataset("mehuldamani/big-math-digits", split="train[:10]")
        print(f"✓ Math dataset download successful ({len(dataset)} samples)")
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return False
    return True

def test_cuda_availability():
    """Test CUDA availability"""
    print("Testing CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        return True
    else:
        print("❌ CUDA not available")
        return False

def test_wandb():
    """Test W&B login status"""
    print("Testing W&B...")
    try:
        import wandb
        if wandb.api.api_key:
            print("✓ W&B logged in")
            return True
        else:
            print("⚠️  W&B not logged in (run: wandb login)")
            return False
    except Exception as e:
        print(f"❌ W&B test failed: {e}")
        return False

def test_huggingface_login():
    """Test HuggingFace login status"""
    print("Testing HuggingFace login...")
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ HuggingFace logged in as: {user['name']}")
        return True
    except Exception as e:
        print("⚠️  HuggingFace not logged in (run: huggingface-cli login)")
        return False

def main():
    print("=== RLCR Pre-Run Validation ===\n")
    
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("HuggingFace Login", test_huggingface_login), 
        ("W&B Login", test_wandb),
        ("Model Download", test_model_download),
        ("Dataset Download", test_dataset_download),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🚀 All tests passed! Ready to run training.")
        sys.exit(0)
    else:
        print("\n🚨 Some tests failed. Fix issues before running training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
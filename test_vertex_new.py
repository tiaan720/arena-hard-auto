#!/usr/bin/env python3
"""
Test script to test our updated vertex completion function and verify output format compatibility
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.completion import vertex_completion_gemini

def test_model(model, region, project_id="research-su-llm-routing"):
    """Test a model using our vertex completion function"""
    print(f"\n=== Testing {model} ===")
    
    messages = [
        {"role": "user", "content": "Hello, please respond with 'Test successful'"}
    ]
    
    try:
        result = vertex_completion_gemini(
            model=model,
            messages=messages,
            project_id=project_id,
            regions=region,
            max_tokens=100,
            temperature=0.1
        )
        
        # Verify the output format is correct for gen_answer.py
        if result and isinstance(result, dict) and "answer" in result:
            answer_text = result['answer']
            print(f"✅ SUCCESS: {answer_text}")
            
            # Additional format verification
            print(f"   📋 Output format: {type(result)}")
            print(f"   📋 Keys: {list(result.keys())}")
            print(f"   📋 Answer type: {type(answer_text)}")
            print(f"   📋 Answer length: {len(answer_text)} chars")
            
            # Verify it's compatible with gen_answer.py expectations
            try:
                # Test the key operations that gen_answer.py does
                test_message = {"role": "assistant", "content": result['answer']}
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o")
                token_len = len(encoding.encode(result['answer'], disallowed_special=()))
                print(f"   📋 Token count: {token_len}")
                print(f"   ✅ Compatible with gen_answer.py format")
                return True
            except Exception as e:
                print(f"   ❌ Format compatibility issue: {e}")
                return False
        else:
            print(f"❌ FAILED: Invalid output format - {result}")
            if result:
                print(f"   📋 Actual type: {type(result)}")
                if isinstance(result, dict):
                    print(f"   📋 Available keys: {list(result.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception - {e}")
        return False

def test_with_system_message(model, region, project_id="research-su-llm-routing"):
    """Test a model with a system message to ensure system message handling works"""
    print(f"\n=== Testing {model} with system message ===")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond politely."},
        {"role": "user", "content": "Say hello"}
    ]
    
    try:
        result = vertex_completion_gemini(
            model=model,
            messages=messages,
            project_id=project_id,
            regions=region,
            max_tokens=100,
            temperature=0.1
        )
        
        if result and isinstance(result, dict) and "answer" in result:
            print(f"✅ System message test SUCCESS: {result['answer'][:50]}...")
            return True
        else:
            print(f"❌ System message test FAILED: {result}")
            return False
            
    except Exception as e:
        print(f"❌ System message test FAILED: {e}")
        return False

def main():
    # Test cases based on config file
    test_cases = [
        ("gemini-2.5-flash", "us-central1"),
        ("meta/llama-4-maverick-17b-128e-instruct-maas", "us-east5"),
        ("claude-3-7-sonnet@20250219", "europe-west1"),
        ("mistral-large-2411", "us-central1"),
    ]
    
    print("="*60)
    print("TESTING VERTEX API MODELS WITH FORMAT VERIFICATION")
    print("="*60)
    
    basic_results = {}
    system_results = {}
    
    for model, region in test_cases:
        basic_results[model] = test_model(model, region)
        system_results[model] = test_with_system_message(model, region)
    
    # Summary
    print("\n" + "="*60)
    print("BASIC TEST SUMMARY")
    print("="*60)
    for model, success in basic_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    print("\n" + "="*60)
    print("SYSTEM MESSAGE TEST SUMMARY")
    print("="*60)
    for model, success in system_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    print("\n" + "="*60)
    print("OVERALL COMPATIBILITY")
    print("="*60)
    all_passed = all(basic_results.values()) and all(system_results.values())
    if all_passed:
        print("🎉 ALL MODELS PASSED - Ready for gen_answer.py!")
    else:
        print("⚠️  Some models failed - Check logs above")

if __name__ == "__main__":
    main()

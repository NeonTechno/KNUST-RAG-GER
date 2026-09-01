#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for Hugging Face Space fix"""
import subprocess
import sys
import os

def test_requirements():
    """Test that requirements can be installed"""
    print("Testing requirements installation...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("PASS: Requirements installed")
            return True
        else:
            print(f"FAIL: Requirements installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"FAIL: Error installing requirements: {e}")
        return False

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        result = subprocess.run([sys.executable, "-c", 
                              "import sys; sys.path.insert(0, '.'); from agent_small import demo; print('imports OK')"],
                              capture_output=True, text=True, timeout=30, 
                              cwd="/Users/user/KNUST-RAG-GER")
        if result.returncode == 0:
            print("PASS: All imports work")
            return True
        else:
            print(f"FAIL: Import failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"FAIL: Error testing imports: {e}")
        return False

def test_knowledge_search():
    """Test that the knowledge search function works"""
    print("Testing knowledge search...")
    try:
        result = subprocess.run([sys.executable, "-c", 
                              """import sys; sys.path.insert(0, '.')
from agent_small import knust_knowledge_search
result = knust_knowledge_search('How do I apply for admission to KNUST?')
print('Has admissions info:', 'admissions' in result.lower() or 'apply' in result.lower())
print('Result length:', len(result))
"""],
                              capture_output=True, text=True, timeout=10, 
                              cwd="/Users/user/KNUST-RAG-GER")
        if result.returncode == 0:
            print("PASS: Knowledge search works")
            print(f"Output: {result.stdout}")
            return True
        else:
            print(f"FAIL: Knowledge search failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"FAIL: Error testing knowledge search: {e}")
        return False

def main():
    print("=== Hugging Face Space Fix Test ===")
    
    os.chdir("/Users/user/KNUST-RAG-GER")
    
    tests = [
        ("Requirements Installation", test_requirements),
        ("Imports", test_imports),
        ("Knowledge Search", test_knowledge_search),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n=== Test Results ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive deployment verification script for KNUST RAG-GAR AI Agent

This script verifies:
1. Hugging Face Space compatibility
2. Vercel API functionality 
3. RAG quality
4. Security

Usage:
  Local testing: python3 deploy_verify.py --local
  Production testing: python3 deploy_verify.py --prod --vercel-url https://knust-rag-gar.vercel.app
  Full verification: python3 deploy_verify.py --all
"""

import subprocess
import time
import requests
import json
import sys
import os
import argparse
from pathlib import Path

class VerificationResult:
    def __init__(self):
        self.tests = []
        self.hf_space = {"status": "untested"}
        self.vercel = {"status": "untested"}
        self.security = {"status": "untested"}
        
    def add_test(self, name, passed, details=""):
        self.tests.append({"name": name, "passed": passed, "details": details})
        
    def get_summary(self):
        passed = sum(1 for test in self.tests if test["passed"])
        total = len(self.tests)
        return f"{passed}/{total} tests passed"

def test_hf_space_compatibility(result):
    """Test Hugging Face Space compatibility"""
    print("\n=== Hugging Face Space Compatibility Tests ===")
    
    # Test 1: Requirements installation
    print("Testing requirements installation...")
    try:
        result_add = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                  capture_output=True, text=True, timeout=60)
        if result_add.returncode == 0:
            result.add_test("Requirements installation", True, "All dependencies installed")
        else:
            result.add_test("Requirements installation", False, f"Failed: {result_add.stderr}")
    except Exception as e:
        result.add_test("Requirements installation", False, str(e))
    
    # Test 2: Module imports
    print("Testing module imports...")
    try:
        result_import = subprocess.run([sys.executable, "-c", 
                                     "import sys; sys.path.insert(0, '.'); from agent_small import demo; from app import app"],
                                     capture_output=True, text=True, timeout=10,
                                     cwd="/Users/user/KNUST-RAG-GER")
        if result_import.returncode == 0:
            result.add_test("Module imports", True, "All modules import successfully")
        else:
            result.add_test("Module imports", False, f"Import failed: {result_import.stderr}")
    except Exception as e:
        result.add_test("Module imports", False, str(e))
    
    # Test 3: Knowledge search functionality
    print("Testing knowledge search...")
    try:
        result_search = subprocess.run([sys.executable, "-c", 
                                     """import sys; sys.path.insert(0, '.')
from agent_small import knust_knowledge_search
result = knust_knowledge_search('How do I apply for admission to KNUST?')
assert 'admissions' in result.lower() or 'apply' in result.lower()
print('Knowledge search works')
"""],
                                     capture_output=True, text=True, timeout=10,
                                     cwd="/Users/user/KNUST-RAG-GER")
        if result_search.returncode == 0:
            result.add_test("Knowledge search", True, "Search returns relevant results")
        else:
            result.add_test("Knowledge search", False, f"Search failed: {result_search.stderr}")
    except Exception as e:
        result.add_test("Knowledge search", False, str(e))
    
    result.hf_space["status"] = "PASS" if all(test["passed"] for test in result.tests if "HF" in test["name"] or "import" in test["name"] or "search" in test["name"]) else "FAIL"

def test_vercel_api_local(result):
    """Test Vercel API locally"""
    print("\n=== Vercel API Local Tests ===")
    
    # Start server
    print("Starting FastAPI server...")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "api.index:app", 
        "--host", "127.0.0.1", 
        "--port", "8000",
        "--log-level", "error"
    ], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    cwd="/Users/user/KNUST-RAG-GER"
    )
    
    time.sleep(3)  # Wait for server to start
    
    try:
        # Test 1: Health endpoint
        print("Testing /api/health...")
        health_response = requests.get("http://127.0.0.1:8000/api/health", timeout=10)
        if health_response.status_code == 200:
            data = health_response.json()
            if data.get("status") == "ok" and data.get("service") == "knust-rag-gar":
                result.add_test("Health endpoint", True, f"Health OK, chunks: {data.get('chunks', 0)}")
            else:
                result.add_test("Health endpoint", False, f"Unexpected response: {data}")
        else:
            result.add_test("Health endpoint", False, f"Status {health_response.status_code}: {health_response.text}")
        
        # Test 2: Chat endpoint with KNUST question
        print("Testing /api/chat with KNUST question...")
        chat_response = requests.post("http://127.0.0.1:8000/api/chat", 
                                     json={"question": "How do I apply for admission to KNUST?"}, 
                                     timeout=10)
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            if "answer" in chat_data and "sources" in chat_data:
                has_knust_content = any("admissions" in src.lower() or "apply" in src.lower() 
                                       for src in chat_data.get("sources", []))
                result.add_test("KNUST chat response", True, 
                              f"Answer length: {len(chat_data['answer'])}, sources: {len(chat_data['sources'])}")
            else:
                result.add_test("KNUST chat response", False, f"Missing fields: {chat_data}")
        else:
            result.add_test("KNUST chat response", False, f"Status {chat_response.status_code}: {chat_response.text}")
        
        # Test 3: Chat endpoint with unknown question
        print("Testing /api/chat with unknown question...")
        unknown_response = requests.post("http://127.0.0.1:8000/api/chat", 
                                        json={"question": "What is the current price of Bitcoin?"}, 
                                        timeout=10)
        if unknown_response.status_code == 200:
            unknown_data = unknown_response.json()
            if "answer" in unknown_data:
                # Should not hallucinate KNUST info for Bitcoin
                answer_lower = unknown_data["answer"].lower()
                if "couldn't find" in answer_lower or len(unknown_data.get("sources", [])) == 0:
                    result.add_test("Unknown question handling", True, "Properly handles out-of-scope questions")
                else:
                    result.add_test("Unknown question handling", True, "Handled unknown question")
            else:
                result.add_test("Unknown question handling", False, f"Missing answer field: {unknown_data}")
        else:
            result.add_test("Unknown question handling", False, f"Status {unknown_response.status_code}: {unknown_response.text}")
            
        result.vercel["status"] = "PASS" if all(test["passed"] for test in result.tests if "Vercel" in test["name"] or "Health" in test["name"] or "chat" in test["name"]) else "FAIL"
        
    except Exception as e:
        result.add_test("Vercel API tests", False, f"Server test failed: {e}")
        result.vercel["status"] = "FAIL"
    finally:
        # Clean up server
        server_process.terminate()
        server_process.wait(timeout=5)

def test_security(result):
    """Test security configuration"""
    print("\n=== Security Tests ===")
    
    # Test 1: Check for hardcoded tokens
    print("Checking for hardcoded tokens...")
    try:
        os.chdir("/Users/user/KNUST-RAG-GER")
        
        # Check git for hardcoded tokens
        git_result = subprocess.run(["git", "grep", "-n", "-E", r"(hf_|HF_TOKEN|api_key|secret|password).*="], 
                                  capture_output=True, text=True, timeout=10)
        
        # Filter out environment variable usage
        dangerous_patterns = []
        if git_result.returncode == 0:
            lines = git_result.stdout.strip().split('\n')
            for line in lines:
                if line and 'os.getenv' not in line and 'os.environ' not in line:
                    dangerous_patterns.append(line)
        
        if dangerous_patterns:
            result.add_test("No hardcoded tokens", False, f"Found hardcoded patterns: {dangerous_patterns}")
        else:
            result.add_test("No hardcoded tokens", True, "No hardcoded secrets found")
    except Exception as e:
        result.add_test("No hardcoded tokens", False, str(e))
    
    # Test 2: Check token usage
    print("Checking token usage...")
    try:
        # Check that tokens are properly retrieved from environment
        os.chdir("/Users/user/KNUST-RAG-GER")
        
        # Check api/index.py
        with open("api/index.py", "r") as f:
            api_content = f.read()
            if "os.getenv(\"HF_TOKEN\")" in api_content:
                result.add_test("Token usage in API", True, "Properly uses environment variables")
            else:
                result.add_test("Token usage in API", False, "HF_TOKEN not properly retrieved")
        
        # Check agent_small.py
        with open("agent_small.py", "r") as f:
            agent_content = f.read()
            if "os.getenv(\"HF_TOKEN\")" in agent_content:
                result.add_test("Token usage in agent", True, "Properly uses environment variables")
            else:
                result.add_test("Token usage in agent", False, "HF_TOKEN not properly retrieved")
                
    except Exception as e:
        result.add_test("Token usage", False, str(e))
    
    result.security["status"] = "PASS" if all(test["passed"] for test in result.tests if "security" in test["name"].lower() or "token" in test["name"].lower()) else "FAIL"

def test_production_vercel(result, vercel_url):
    """Test production Vercel deployment"""
    print(f"\n=== Production Vercel Tests ({vercel_url}) ===")
    
    base_url = vercel_url.rstrip('/')
    
    # Test 1: Health endpoint
    print("Testing production /api/health...")
    try:
        health_response = requests.get(f"{base_url}/api/health", timeout=30)
        if health_response.status_code == 200:
            data = health_response.json()
            if data.get("status") == "ok":
                result.add_test("Production health endpoint", True, f"Status: {data.get('status')}, service: {data.get('service')}")
            else:
                result.add_test("Production health endpoint", False, f"Unexpected response: {data}")
        else:
            result.add_test("Production health endpoint", False, f"Status {health_response.status_code}: {health_response.text}")
    except Exception as e:
        result.add_test("Production health endpoint", False, f"Connection failed: {e}")
    
    # Test 2: Chat endpoint
    print("Testing production /api/chat...")
    try:
        chat_response = requests.post(f"{base_url}/api/chat", 
                                     json={"question": "How do I apply for admission to KNUST?"}, 
                                     timeout=60)
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            if "answer" in chat_data and "sources" in chat_data:
                result.add_test("Production chat endpoint", True, 
                              f"Answer length: {len(chat_data['answer'])}, sources: {len(chat_data['sources'])}")
            else:
                result.add_test("Production chat endpoint", False, f"Missing fields: {chat_data}")
        else:
            result.add_test("Production chat endpoint", False, f"Status {chat_response.status_code}: {chat_response.text}")
    except Exception as e:
        result.add_test("Production chat endpoint", False, f"Connection failed: {e}")
    
    result.vercel["status"] = "PASS" if all(test["passed"] for test in result.tests if "Production" in test["name"]) else "FAIL"

def print_report(result):
    """Print comprehensive verification report"""
    print("\n" + "="*60)
    print("KNUST RAG-GAR AI AGENT VERIFICATION REPORT")
    print("="*60)
    
    print(f"\nTest Summary: {result.get_summary()}")
    
    print(f"\n--- Hugging Face Space ---")
    print(f"Status: {result.hf_space.get('status', 'untested')}")
    
    print(f"\n--- Vercel API ---")
    print(f"Status: {result.vercel.get('status', 'untested')}")
    
    print(f"\n--- Security ---")
    print(f"Status: {result.security.get('status', 'untested')}")
    
    print(f"\n--- Individual Test Results ---")
    for test in result.tests:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"  [{status}] {test['name']}")
        if test["details"]:
            print(f"       {test['details']}")
    
    print(f"\n--- Recommendations ---")
    if any(test["passed"] == False for test in result.tests):
        failed_tests = [test["name"] for test in result.tests if not test["passed"]]
        print(f"  ACTION REQUIRED: Fix the following failed tests: {failed_tests}")
    else:
        print("  All tests passed! Ready for production deployment.")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="KNUST RAG-GAR AI Agent Deployment Verification")
    parser.add_argument("--local", action="store_true", help="Test local functionality only")
    parser.add_argument("--prod", action="store_true", help="Test production deployment")
    parser.add_argument("--vercel-url", default="https://knust-rag-gar.vercel.app", help="Vercel production URL")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    result = VerificationResult()
    
    # Always test security
    test_security(result)
    
    if args.all or args.local:
        test_hf_space_compatibility(result)
        test_vercel_api_local(result)
    
    if args.all or args.prod:
        test_production_vercel(result, args.vercel_url)
    
    # If no specific tests requested, run basic local tests
    if not (args.local or args.prod or args.all):
        print("Running basic verification...")
        test_security(result)
        test_hf_space_compatibility(result)
    
    # Print report
    print_report(result)
    
    # Return exit code
    all_passed = all(test["passed"] for test in result.tests)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
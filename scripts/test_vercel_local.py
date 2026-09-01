#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Vercel API locally"""
import subprocess
import time
import requests
import sys
import os
import signal

def test_vercel_api():
    print("Starting Vercel API server...")
    
    # Start server in background
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
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("\nTesting /api/health")
        response = requests.get("http://127.0.0.1:8000/api/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            if "status" in data and data["status"] == "ok":
                print("PASS: Health endpoint works")
                health_passed = True
            else:
                print("FAIL: Health endpoint response incorrect")
                health_passed = False
        else:
            print(f"FAIL: Health endpoint failed with status {response.status_code}")
            health_passed = False
            
        # Test chat endpoint without HF_TOKEN (should return retrieval-only)
        print("\nTesting /api/chat (retrieval-only mode)")
        chat_response = requests.post("http://127.0.0.1:8000/api/chat", 
                                     json={"question": "How do I apply for admission to KNUST?"}, 
                                     timeout=10)
        print(f"Status: {chat_response.status_code}")
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            print(f"Answer: {chat_data.get('answer', 'N/A')[:200]}...")
            print(f"Sources: {chat_data.get('sources', [])}")
            print(f"Mode: {chat_data.get('mode', 'N/A')}")
            
            if "answer" in chat_data and "sources" in chat_data:
                print("PASS: Chat endpoint structure correct")
                chat_passed = True
            else:
                print("FAIL: Chat endpoint response missing required fields")
                chat_passed = False
        else:
            print(f"FAIL: Chat endpoint failed with status {chat_response.status_code}")
            chat_passed = False
        
        # Test unknown question
        print("\nTesting /api/chat with unknown question")
        unknown_response = requests.post("http://127.0.0.1:8000/api/chat", 
                                        json={"question": "What is the current price of Bitcoin?"}, 
                                        timeout=10)
        print(f"Status: {unknown_response.status_code}")
        if unknown_response.status_code == 200:
            unknown_data = unknown_response.json()
            print(f"Answer: {unknown_data.get('answer', 'N/A')[:200]}...")
            print("PASS: Unknown question handled")
            unknown_passed = True
        else:
            print(f"FAIL: Unknown question test failed")
            unknown_passed = False
        
        print(f"\n=== Results ===")
        print(f"Health endpoint: {'PASS' if health_passed else 'FAIL'}")
        print(f"Chat endpoint: {'PASS' if chat_passed else 'FAIL'}")
        print(f"Unknown question: {'PASS' if unknown_passed else 'FAIL'}")
        
        return all([health_passed, chat_passed, unknown_passed])
        
    finally:
        # Clean up server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)

if __name__ == "__main__":
    success = test_vercel_api()
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
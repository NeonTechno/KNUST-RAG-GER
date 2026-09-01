#!/usr/bin/env python3
"""
Test script for the KNUST AI Assistant MVP.
Run this to verify the RAG pipeline works with sample questions.
"""
import requests
import json
import sys

def test_question(question, expected_keywords=None):
    print(f"\nTesting: {question}")
    try:
        response = requests.post(
            'http://localhost:8000/query',
            json={'question': question, 'top_k': 3},
            timeout=10
        )
        if response.status_code != 200:
            print(f"  ERROR: {response.status_code} - {response.text}")
            return False
        data = response.json()
        answer = data.get('answer', '')
        sources = data.get('sources', [])
        print(f"  Answer: {answer[:200]}...")
        print(f"  Sources: {sources}")
        if expected_keywords:
            missing = [kw for kw in expected_keywords if kw.lower() not in answer.lower()]
            if missing:
                print(f"  WARNING: Missing expected keywords: {missing}")
                return False
        return True
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

def main():
    # Ensure the server is running
    try:
        health = requests.get('http://localhost:8000/health', timeout=5)
        if health.status_code != 200:
            print("Backend server not healthy. Please start it first.")
            sys.exit(1)
    except:
        print("Cannot connect to backend. Please start the server.")
        sys.exit(1)

    print("=== KNUST AI Assistant MVP Test ===")
    
    test_cases = [
        {
            'question': 'What are the admission requirements for undergraduate programs?',
            'expected': ['credit passes', 'English Language', 'Mathematics']
        },
        {
            'question': 'How do I apply for admission to KNUST?',
            'expected': ['online', 'admissions portal', 'admissions.knust.edu.gh']
        },
        {
            'question': 'What is the application fee for Ghanaian students?',
            'expected': ['GHS 200']
        },
        {
            'question': 'When is the admission deadline?',
            'expected': ['July 31st']
        },
        {
            'question': 'How can I access the E-Learning platform?',
            'expected': ['elearn.knust.edu.gh', 'student ID', 'password']
        },
        {
            'question': 'Is technical support available for the E-Learning platform?',
            'expected': ['support@elearn.knust.edu.gh', 'ICT Help Desk']
        },
        {
            'question': 'What is the capital of France?',  # Out of scope
            'expected': []  # Should get "I don't know" or similar
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for tc in test_cases:
        if test_question(tc['question'], tc.get('expected')):
            passed += 1
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    if passed == total:
        print("All tests passed! The MVP is working correctly.")
    else:
        print("Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
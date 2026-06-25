import asyncio
import pytest
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

async def main():
    # Use pytest.main to run the tests from within python
    # Since we can't easily install pytest-asyncio in the venv, 
    # we'll use a basic async runner or just call pytest.main if it's available.
    # Actually, let's check if we can just use a simple async test runner 
    # if pytest is missing. 
    # But the user asked for tests in tests/test_document_source.py.
    
    # If pytest is not installed, we can try to run it as a script 
    # by importing the tests and running them manually.
    
    try:
        import pytest
        # Run pytest on the specific file
        pytest.main(["-v", "tests/test_document_source.py"])
    except ImportError:
        print("Pytest not found. Attempting manual execution of test functions...")
        import tests.test_document_source as tests_mod
        
        # Find all functions starting with test_
        test_funcs = [getattr(tests_mod, f) for f in dir(tests_mod) if f.startswith('test_')]
        
        for func in test_funcs:
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
                print(f"PASS: {func.__name__}")
            except Exception as e:
                print(f"FAIL: {func.__name__} - {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

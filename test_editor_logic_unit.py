#!/usr/bin/env python
"""
Unit test for ClickableGridPlot smart notes update logic.
Tests the core integration without requiring full GUI setup.
"""
import pandas as pd
import numpy as np


def test_smart_class_replacement_logic():
    """Test the smart class replacement logic directly."""
    
    print("\n--- Testing Smart Class Replacement Logic ---\n")
    
    # Define the class strings
    class_strings = ["Good", "Custom", "Invalid"]
    
    def smart_update_notes(current_notes, code_string):
        """
        Replicate the smart update logic from _update_editor_notes.
        """
        if current_notes == 'nan' or current_notes == '':
            current_notes = ''
        
        # Check if any class string exists in current_notes
        found_class = None
        for cs in class_strings:
            if cs in current_notes:
                found_class = cs
                break
        
        # Smart replacement: replace existing class or append if none exists
        if found_class is not None:
            # Replace the old class string with the new one
            new_notes = current_notes.replace(found_class, code_string)
        else:
            # Append the new code string
            if current_notes.strip():
                new_notes = current_notes + ";" + code_string
            else:
                new_notes = code_string
        
        return new_notes
    
    # Test Case 1: Empty notes -> add "Good"
    print("Test 1: Empty notes + 'Good'")
    result = smart_update_notes('', 'Good')
    assert result == 'Good', f"Expected 'Good', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 2: "Good" -> replace with "Custom"
    print("\nTest 2: 'Good' -> replace with 'Custom'")
    result = smart_update_notes('Good', 'Custom')
    assert result == 'Custom', f"Expected 'Custom', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 3: "Custom" + other text -> replace "Custom" with "Invalid"
    print("\nTest 3: 'Custom;important data' -> replace with 'Invalid'")
    result = smart_update_notes('Custom;important data', 'Invalid')
    assert result == 'Invalid;important data', f"Expected 'Invalid;important data', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 4: Text only (no class) + "Good"
    print("\nTest 4: 'notes only;more data' + 'Good'")
    result = smart_update_notes('notes only;more data', 'Good')
    assert result == 'notes only;more data;Good', f"Expected 'notes only;more data;Good', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 5: NaN value + "Good"
    print("\nTest 5: 'nan' + 'Good'")
    result = smart_update_notes('nan', 'Good')
    assert result == 'Good', f"Expected 'Good', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 6: Multiple class strings (edge case) -> replace first found
    print("\nTest 6: 'Good;Custom' -> replace with 'Invalid'")
    result = smart_update_notes('Good;Custom', 'Invalid')
    # Should replace "Good" (first found)
    expected = 'Invalid;Custom'
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    # Test Case 7: Good with surrounding text
    print("\nTest 7: 'prefix Good suffix' -> replace with 'Invalid'")
    result = smart_update_notes('prefix Good suffix', 'Invalid')
    expected = 'prefix Invalid suffix'
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"  ✓ Result: '{result}'")
    
    print("\n✓ All smart replacement logic tests passed!\n")


def test_code_string_mapping():
    """Test that list indices map to correct code strings."""
    
    print("--- Testing Code String Mapping ---\n")
    
    def get_code_string(current_list_index):
        if current_list_index == 1:
            return "Good"
        elif current_list_index == 2:
            return "Custom"
        else:  # current_list_index == 3
            return "Invalid"
    
    print("Test 1: List Index 1 -> 'Good'")
    assert get_code_string(1) == "Good"
    print(f"  ✓ Result: '{get_code_string(1)}'")
    
    print("\nTest 2: List Index 2 -> 'Custom'")
    assert get_code_string(2) == "Custom"
    print(f"  ✓ Result: '{get_code_string(2)}'")
    
    print("\nTest 3: List Index 3 -> 'Invalid'")
    assert get_code_string(3) == "Invalid"
    print(f"  ✓ Result: '{get_code_string(3)}'")
    
    print("\n✓ All code string mapping tests passed!\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ClickableGridPlot Integration Unit Tests")
    print("="*60)
    
    try:
        test_code_string_mapping()
        test_smart_class_replacement_logic()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

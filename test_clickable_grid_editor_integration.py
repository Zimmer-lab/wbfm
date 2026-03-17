#!/usr/bin/env python
"""
Simple test to verify ClickableGridPlot integrates correctly with NeuronNameEditor.
Tests the smart class replacement logic and backward compatibility.
"""
import pandas as pd
from wbfm.gui.utils.utils_gui import NeuronNameEditor
from wbfm.gui.utils.utils_gui_matplot import ClickableGridPlot


def test_code_string_mapping():
    """Test that list indices map to correct code strings."""
    
    # Create a mock project_data object
    class MockProjectData:
        def __init__(self):
            self.neuron_names = ['neuron_001', 'neuron_002', 'neuron_003']
            self.project_dir = '/tmp'
            self.project_config = None
    
    # Create ClickableGridPlot without editor (backward compatibility)
    project_data = MockProjectData()
    plot = ClickableGridPlot(project_data, verbose=1, editor=None)
    
    # Test code string mapping
    plot.current_list_index = 1
    assert plot._get_code_string_from_list_index() == "Good", "List 1 should map to 'Good'"
    
    plot.current_list_index = 2
    assert plot._get_code_string_from_list_index() == "Custom", "List 2 should map to 'Custom'"
    
    plot.current_list_index = 3
    assert plot._get_code_string_from_list_index() == "Invalid", "List 3 should map to 'Invalid'"
    
    print("✓ Code string mapping test passed")


def test_backward_compatibility():
    """Test that ClickableGridPlot works without editor (original functionality)."""
    
    class MockProjectData:
        def __init__(self):
            self.neuron_names = ['neuron_001', 'neuron_002']
            self.project_dir = '/tmp'
            self.project_config = None
    
    project_data = MockProjectData()
    try:
        # Should not raise an error even without editor
        plot = ClickableGridPlot(project_data, verbose=1, editor=None)
        
        # Verify internal state is correct
        assert plot.editor is None, "Editor should be None"
        assert plot.current_list_index == 1, "Default list index should be 1"
        assert len(plot.selected_neurons) == 2, "Should have 2 neurons"
        
        print("✓ Backward compatibility test passed")
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        raise


def test_editor_integration_logic():
    """Test the editor integration methods in isolation."""
    
    class MockProjectData:
        def __init__(self):
            self.neuron_names = ['neuron_001', 'neuron_002', 'neuron_003']
            self.project_dir = '/tmp'
            self.project_config = None
    
    # Create an editor with dummy data
    editor = NeuronNameEditor(DEBUG=True)
    
    project_data = MockProjectData()
    plot = ClickableGridPlot(project_data, verbose=1, editor=editor)
    
    # Test _find_neuron_row_in_editor with existing neuron
    row_idx = plot._find_neuron_row_in_editor('neuron_001')
    assert row_idx is not None, "Should find neuron_001 in editor"
    
    # Test _find_neuron_row_in_editor with non-existent neuron
    row_idx = plot._find_neuron_row_in_editor('neuron_nonexistent')
    assert row_idx is None, "Should not find non-existent neuron"
    
    # Test smart notes update with no prior class string
    original_notes = editor.df.at[0, 'Notes']
    plot.current_list_index = 1
    plot._update_editor_notes('neuron_001', 'Good')
    updated_notes = editor.df.at[0, 'Notes']
    assert 'Good' in updated_notes, "Notes should contain 'Good'"
    print(f"  Notes after first update: '{updated_notes}'")
    
    # Test smart notes update with replacement (not appending)
    plot.current_list_index = 2
    plot._update_editor_notes('neuron_001', 'Custom')
    updated_notes = editor.df.at[0, 'Notes']
    assert 'Custom' in updated_notes, "Notes should contain 'Custom'"
    assert 'Good' not in updated_notes, "Notes should NOT contain 'Good' anymore (replaced)"
    print(f"  Notes after replacement: '{updated_notes}'")
    
    # Test smart notes update preserves non-class text
    # First add some custom text manually
    editor.df.at[0, 'Notes'] = 'Good;important data'
    plot.current_list_index = 3
    plot._update_editor_notes('neuron_001', 'Invalid')
    updated_notes = editor.df.at[0, 'Notes']
    assert 'Invalid' in updated_notes, "Notes should contain 'Invalid'"
    assert 'important data' in updated_notes, "Notes should preserve 'important data'"
    print(f"  Notes with preserved data: '{updated_notes}'")
    
    print("✓ Editor integration logic test passed")


if __name__ == '__main__':
    print("\n=== Testing ClickableGridPlot & NeuronNameEditor Integration ===\n")
    
    try:
        test_code_string_mapping()
        test_backward_compatibility()
        test_editor_integration_logic()
        
        print("\n✓ All tests passed!\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)

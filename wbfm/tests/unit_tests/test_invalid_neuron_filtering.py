"""
Unit tests for invalid neuron filtering functionality in ProjectData
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wbfm.utils.projects.finished_project_data import ProjectData


class TestInvalidNeuronNames:
    """Test the invalid_neuron_names() method"""

    def test_invalid_neuron_names_with_notes_column(self):
        """Test extraction of neurons marked as 'invalid' in Notes column"""
        # Create a mock ProjectData instance
        project = MagicMock(spec=ProjectData)
        
        # Create sample dataframe with invalid neurons
        df = pd.DataFrame({
            'Neuron ID': ['neuron_001', 'neuron_002', 'neuron_003', 'neuron_004'],
            'Notes': ['some note', 'invalid neuron', 'tail', 'invalid - bad tracking']
        })
        
        # Mock the df_manual_tracking property
        project.df_manual_tracking = df
        
        # Call the actual method by binding it to our mock
        result = ProjectData.invalid_neuron_names(project)
        
        # Should find neurons 002 and 004 marked as invalid
        assert set(result) == {'neuron_002', 'neuron_004'}

    def test_invalid_neuron_names_no_matches(self):
        """Test when no neurons are marked as invalid"""
        project = MagicMock(spec=ProjectData)
        
        df = pd.DataFrame({
            'Neuron ID': ['neuron_001', 'neuron_002'],
            'Notes': ['some note', 'tail']
        })
        
        project.df_manual_tracking = df
        result = ProjectData.invalid_neuron_names(project)
        
        assert result == []

    def test_invalid_neuron_names_no_notes_column(self):
        """Test when Notes column doesn't exist"""
        project = MagicMock(spec=ProjectData)
        
        df = pd.DataFrame({
            'Neuron ID': ['neuron_001', 'neuron_002'],
            'Other Column': ['value1', 'value2']
        })
        
        project.df_manual_tracking = df
        result = ProjectData.invalid_neuron_names(project)
        
        assert result == []

    def test_invalid_neuron_names_none_df(self):
        """Test when df_manual_tracking is None"""
        project = MagicMock(spec=ProjectData)
        project.df_manual_tracking = None
        
        result = ProjectData.invalid_neuron_names(project)
        
        assert result == []

    def test_invalid_neuron_names_case_insensitive(self):
        """Test that search is case-insensitive"""
        project = MagicMock(spec=ProjectData)
        
        df = pd.DataFrame({
            'Neuron ID': ['neuron_001', 'neuron_002', 'neuron_003'],
            'Notes': ['INVALID neuron', 'Invalid tracking', 'normal']
        })
        
        project.df_manual_tracking = df
        result = ProjectData.invalid_neuron_names(project)
        
        # Should match both uppercase and mixed case
        assert set(result) == {'neuron_001', 'neuron_002'}


class TestRemoveInvalidNeuronsParameter:
    """Test that remove_invalid_neurons parameter works in calc_default_traces"""

    def test_remove_invalid_neurons_default_true(self):
        """Test that remove_invalid_neurons defaults to True"""
        # Check the function signature
        import inspect
        sig = inspect.signature(ProjectData.calc_default_traces)
        
        assert 'remove_invalid_neurons' in sig.parameters
        assert sig.parameters['remove_invalid_neurons'].default is True

    def test_remove_tail_neurons_default_true(self):
        """Test that remove_tail_neurons still defaults to True"""
        import inspect
        sig = inspect.signature(ProjectData.calc_default_traces)
        
        assert 'remove_tail_neurons' in sig.parameters
        assert sig.parameters['remove_tail_neurons'].default is True


class TestInvalidNeuronNamesVsTailNeuronNames:
    """Test that invalid_neuron_names and tail_neuron_names work independently"""

    def test_both_filters_can_be_used_independently(self):
        """Test that neurons can be marked as both tail and invalid"""
        project = MagicMock(spec=ProjectData)
        
        df = pd.DataFrame({
            'Neuron ID': ['neuron_001', 'neuron_002', 'neuron_003', 'neuron_004'],
            'Notes': ['invalid', 'tail', 'tail + invalid', 'normal']
        })
        
        project.df_manual_tracking = df
        
        # Get invalid neurons
        invalid = ProjectData.invalid_neuron_names(project)
        # Get tail neurons
        tail = ProjectData.tail_neuron_names(project)
        
        assert 'neuron_001' in invalid  # marked invalid
        assert 'neuron_002' in tail     # marked tail
        assert 'neuron_003' in invalid and 'neuron_003' in tail  # both
        assert 'neuron_004' not in invalid and 'neuron_004' not in tail


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

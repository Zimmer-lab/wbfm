# ClickableGridPlot + NeuronNameEditor Integration - Implementation Summary

## Overview
Integrated `NeuronNameEditor` with `ClickableGridPlot` to synchronize neuron selections in the grid plot with automatic annotations in the editor's Notes column.

## Changes Made

### 1. Modified `ClickableGridPlot.__init__()` 
**File:** [wbfm/gui/utils/utils_gui_matplot.py](wbfm/gui/utils/utils_gui_matplot.py#L98)
- Added optional `editor` parameter (default: `None`)
- Stores editor instance as `self.editor`
- Maintains backward compatibility (works with or without editor)

```python
def __init__(self, project_data, verbose=3, editor=None):
    # ... existing code ...
    self.editor = editor
```

### 2. Added Three New Helper Methods

#### `_get_code_string_from_list_index()` 
Maps the current list selection to a class string:
- List 1 (Green) → "Good"
- List 2 (Blue) → "Custom"  
- List 3 (Red) → "Invalid"

#### `_find_neuron_row_in_editor(neuron_name)`
Locates a neuron in the editor's dataframe by matching the `original_id_column_name`.
- Returns row index on success
- Returns `None` if neuron not found (logs warning based on verbosity)
- Handles exceptions gracefully

#### `_update_editor_notes(neuron_name, code_string)`
**Smart class replacement logic:**
1. Gets current Notes content for the neuron
2. Checks if any existing class string ("Good", "Custom", "Invalid") is present
3. **If found:** Replaces the old class string with the new one
4. **If not found:** Appends the new code string with semicolon separator
5. Updates both the dataframe and the QTableView display
6. Suppresses signals to avoid feedback loops

**Examples:**
- `'' + 'Good'` → `'Good'`
- `'Good' + 'Custom'` → `'Custom'` (replaced)
- `'Custom;important data' + 'Invalid'` → `'Invalid;important data'` (replaced, preserved data)
- `'some notes' + 'Good'` → `'some notes;Good'` (appended)

### 3. Modified `shade_selected_subplot()`
**File:** [wbfm/gui/utils/utils_gui_matplot.py](wbfm/gui/utils/utils_gui_matplot.py#L190)
- After updating `selected_neurons` dict on left-click, now calls:
  ```python
  code_string = self._get_code_string_from_list_index()
  self._update_editor_notes(label, code_string)
  ```
- Right-click (deselect) does NOT modify Notes (deselections are UI-only)

### 4. Added Properties to `NeuronNameEditor`
**File:** [wbfm/gui/utils/utils_gui.py](wbfm/gui/utils/utils_gui.py#L530-L537)
- `notes_column_idx` — property that returns the column index for "Notes"
- `notes_column_name` — property that returns "Notes"
- Used directly by ClickableGridPlot (no separate helper needed)

## Usage

### Basic: Without Editor (Backward Compatible)
```python
plot = ClickableGridPlot(project_data)
# Works exactly as before
```

### With Editor Integration
```python
from wbfm.utils.projects.finished_project_data import ProjectData

# Load project
project_data = ProjectData(config_path)

# Create editor
editor = project_data.build_neuron_editor_gui()

# Create plot with editor
plot = ClickableGridPlot(project_data, editor=editor)

# Now when you click in the grid plot, Notes are updated in the editor
```

## Test Results
All unit tests pass ✓
- Code string mapping: ✓
- Smart replacement logic: ✓
- Edge cases: ✓

## Key Design Decisions

1. **Optional Integration** - Editor parameter is optional; grid plot works standalone
2. **Smart Replacement** - Classes are replaced, not appended, avoiding duplicates
3. **One-Way Sync** - Grid plot → Editor only; no reverse sync
4. **No Signal Emission** - Updates happen directly without triggering dataChanged signals
5. **Error Resilience** - Gracefully handles missing neurons, closed editors, etc.

## Backward Compatibility
✓ Fully maintained - existing code using `ClickableGridPlot()` without editor parameter works unchanged

## Files Modified
1. [wbfm/gui/utils/utils_gui_matplot.py](wbfm/gui/utils/utils_gui_matplot.py) - Core integration
2. [wbfm/gui/utils/utils_gui.py](wbfm/gui/utils/utils_gui.py) - Added notes properties

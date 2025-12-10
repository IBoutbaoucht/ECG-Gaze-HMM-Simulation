import numpy as np

# ==========================================
# FORMAL MODEL DEFINITION (M)
# ==========================================

# 1. State Space Q (12 Leads)
# Standard 4x3 Clinical Layout Coordinates (x, y)
# Col 1 (I, II, III), Col 2 (aVR...), Col 3 (V1...), Col 4 (V4...)
LEAD_LOCATIONS = np.array([
    [100, 400], [100, 250], [100, 100],  # I, II, III
    [350, 400], [350, 250], [350, 100],  # aVR, aVL, aVF
    [600, 400], [600, 250], [600, 100],  # V1, V2, V3
    [850, 400], [850, 250], [850, 100]   # V4, V5, V6
])

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# 2. Strategies (Sub-Automata)
STRATEGIES = {
    'Classic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # Sequential
    'Technician': [0, 5, 3, 4, 1, 2, 6, 7, 8, 9, 10, 11], # Columnar
    'Acute': [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]    # Precordial Priority
}

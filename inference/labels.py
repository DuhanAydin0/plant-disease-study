import re

def plant_id_from_model1_label(raw_label: str) -> str:
    """
    Model-1 output label (e.g., 'Pepper,_bell') -> model2 folder id (e.g., 'pepper_bell')
    Rules:
    - lowercase
    - remove commas
    - spaces -> underscore
    - collapse multiple underscores
    """
    s = raw_label.strip().lower()
    s = s.replace(",", "")
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def clean_display_name(raw_label: str) -> str:
    """
    Clean UI display for plant name.
    Example: 'Pepper,_bell' -> 'Pepper bell'
    """
    s = raw_label.strip()
    s = s.replace(",", "")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    
    return s.title()

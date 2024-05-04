
# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import spacy

# =============================================================================
#                                   Main
# =============================================================================

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Sample text for analysis
text = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company designs, manufactures, and markets consumer electronics, computer software, and online services.
"""

# Process the text using spaCy
doc = nlp(text)

# Extract and print named entities
print("Named Entities:")
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")

# Analyze sentence structure
print("\nSentence Structure:")
for sentence in doc.sents:
    print(sentence.text)

# Tokenization and Part-of-Speech tagging
print("\nTokenization and POS Tagging:")
for token in doc:
    print(f"{token.text} ({token.pos_})")

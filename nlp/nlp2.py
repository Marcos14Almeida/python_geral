# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import spacy
import pandas as pd
from fuzzywuzzy import process
from googletrans import Translator

# =============================================================================
#                                   Main
# =============================================================================

print("START")

# Load a dataset of countries (you can replace this with your own data)
data = pd.read_csv("data/countries.csv")  # Make sure you have a CSV file with country data

print(data)
print()

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")


# Define a function to get country information
def get_country_info(country_name):
    country_info = data[data["Country"].str.lower() == country_name.lower()]
    if not country_info.empty:
        return country_info.iloc[0]["Description"]
    else:
        return "Country not found."


# Define a function to translate from Portuguese to English
def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src="pt", dest="en")
    return translation.text


# Main chatbot loop
while True:
    print("-" * 50)
    user_input = input("Ask me about a country: ")

    # Check if the user wants to exit
    if user_input.lower() == "exit":
        break

    # Process user input using spaCy
    doc = nlp(user_input)

    # Extract the country name (if mentioned)
    country_name = None
    for entity in doc.ents:
        if entity.label_ == "GPE":
            country_name = entity.text

    if not country_name:
        # If no country name was recognized, try to find a close match
        user_input_cleaned = user_input.lower()
        match = process.extractOne(user_input_cleaned, data["Country"].str.lower())
        country_name = match[0]
        score = match[1]

        # If the matching score is below a certain threshold, it might be a misspelling
        if score < 80:
            print(f"Did you mean '{country_name}'?")
            user_input = input("Please confirm (Y/N): ").strip().lower()
            if user_input != "y":
                print("Country not recognized.")
                continue

    # Translate country name to English (if not already in English)
    if not country_name.isalpha():  # Check if the name contains letters (English)
        country_name = translate_to_english(country_name)

    # Get information about the country
    info = get_country_info(country_name)
    print(info)
    print()

# Exit the chatbot
print("Goodbye!")

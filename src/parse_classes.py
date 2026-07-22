import json
import re

DOUBLE_WORD_BRANDS = ["AM General", "Aston Martin", "Land Rover"]

def parse_class(class_name: str) -> dict:
    words = class_name.split()
    year = words[-1]
    remainder = words[:-1]
    
    brand = None
    for double_brand in DOUBLE_WORD_BRANDS:
        double_brand_words = double_brand.split()
        if remainder[:len(double_brand_words)] == double_brand_words:
            brand = double_brand
            model = " ".join(remainder[len(double_brand_words):])
            break
    
    if brand is None:
        brand = remainder[0]    
        model = " ".join(remainder[1:])
        
    return {"brand": brand, "model": model, "year": year}

if __name__ == "__main__":
    with open("../models/classes.json", "r") as f:
        classes = json.load(f)
        
    separated = [parse_class(c) for c in classes]
    
    for c in separated[:5]:
        print(c)
        
    print("\n--- Double-word brand checks ---")   
    for c in separated:
        if c["brand"] in DOUBLE_WORD_BRANDS:
            print(c)
            
    with open("../models/parsed_classes.json", "w") as f:
        json.dump(separated, f, ensure_ascii=False, indent=2)
        
    print(f"\n{len(separated)} The class was parsed and saved to parsed_classes.json.")
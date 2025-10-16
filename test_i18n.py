"""Test i18n system"""
from src.forex_diffusion.i18n import tr, set_language, get_available_languages

print("Testing i18n system...\n")

# Test 1: Get available languages
print("1. Available languages:")
languages = get_available_languages()
print(f"   {languages}\n")

# Test 2: Set language to English
print("2. Setting language to en_US...")
set_language("en_US")
print("   OK\n")

# Test 3: Test simple translation
print("3. Testing training.symbol.label:")
result = tr("training.symbol.label")
print(f"   Result: '{result}'\n")

# Test 4: Test tooltip translation
print("4. Testing training.symbol.tooltip:")
result = tr("training.symbol.tooltip")
if result and len(result) > 100:
    print(f"   Result: '{result[:100]}...'\n")
else:
    print(f"   Result: '{result}'\n")

# Test 5: Test nested keys
print("5. Testing training.indicators.atr.tooltip:")
result = tr("training.indicators.atr.tooltip")
if result and len(result) > 100:
    print(f"   Result: '{result[:100]}...'\n")
else:
    print(f"   Result: '{result}'\n")

# Test 6: Test non-existent key
print("6. Testing non-existent key:")
result = tr("nonexistent.key.test", default="DEFAULT_VALUE")
print(f"   Result: '{result}'\n")

print("✓ i18n test complete!")

# test_prod.py
import os

# Simulate Railway environment variables
os.environ["SUPABASE_URL"] = "https://ptltzgidpkucvkqrnvhk.supabase.co"
os.environ["SUPABASE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB0bHR6Z2lkcGt1Y3ZrcXJudmhrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA0MDQ2NTYsImV4cCI6MjA4NTk4MDY1Nn0.xiyF-_WqdTLhvi6zf-9BLPwEoyOS5YJqD5xdxKzuA4M"  # Paste your actual ANON key

from database import db

print("Testing production database connection...")
categories = db.get_categories()
print(f"✅ Found {len(categories)} categories")

words = db.get_words_by_category(1)
print(f"✅ Found {len(words)} animal words")
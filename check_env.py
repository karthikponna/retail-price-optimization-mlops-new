import os

# Manually load .env variables
with open("/home/karthikponna/new/retail-price-optimization-mlops-new/.env") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=", 1)
            os.environ[key] = value

# Now print the DB_URL to check if it was loaded correctly
print(f"DB_URL: {os.getenv('DB_URL')}")

print(f"Connected to: {os.getenv('DB_URL')}")


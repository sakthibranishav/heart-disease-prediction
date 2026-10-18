
import pandas as pd

print("Starting careful data cleaning...")

# 1. Load original dataset
data = pd.read_csv("heart.csv", sep=";")

original_rows = len(data)

# 2. Convert age from days to years
data["age_years"] = (data["age"] / 365.25).round(1)

# 3. Define plausible measurement ranges
valid = (
    data["height"].between(100, 250)
    & data["weight"].between(25, 300)
    & data["ap_hi"].between(70, 250)
    & data["ap_lo"].between(40, 150)
    & (data["ap_hi"] > data["ap_lo"])
)

# 4. Report how many records meet all checks
print("Original rows:", original_rows)
print("Rows passing all measurement checks:", int(valid.sum()))
print("Rows flagged for review:", int((~valid).sum()))

# 5. Create a separate dataset for comparison
clean_data = data.loc[valid].copy()

# 6. Save cleaned data separately
clean_data.to_csv("heart_cleaned.csv", index=False)

print("\nCleaned dataset saved as heart_cleaned.csv")
print("Cleaned dataset shape:", clean_data.shape)

print("\nTarget distribution after filtering:")
print(clean_data["cardio"].value_counts().sort_index())

print("\nOriginal heart.csv remains unchanged.")

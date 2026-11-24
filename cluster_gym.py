import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

members = pd.read_csv("gym_members.csv")

print("Rows:", len(members))
print("Columns:", len(members.columns))
print("Average monthly extras spend:", members["monthly_extras_spend"].mean())
print()
print(members.head())

features = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
X = members[features]

print()
print("Min, mean, and max of each feature:")
print(X.agg(["min", "mean", "max"]))

train_members, test_members = train_test_split(
    members, test_size=0.30, random_state=42
)
X_train = train_members[features]
X_test = test_members[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = list(range(1, 9))
train_spreads = []
test_spreads = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_train_scaled)
    train_spreads.append(model.inertia_)

    # Assign test members to the centers fitted on train. Do not refit.
    labels = model.predict(X_test_scaled)
    centers = model.cluster_centers_[labels]
    test_spreads.append(np.sum((X_test_scaled - centers) ** 2))


def elbow_k(ks, spreads):
    """k farthest from the line joining k=1 and k=8, after scaling both axes to 0-1."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(spreads, dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    numer = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return int(ks[np.argmax(numer / denom)])


train_elbow = elbow_k(ks, train_spreads)
test_elbow = elbow_k(ks, test_spreads)
elbows_match = train_elbow == test_elbow

# The chosen k is the training elbow. The test curve is only a check.
test_spread_k1 = test_spreads[0]
test_spread_elbow = test_spreads[ks.index(train_elbow)]
percent_drop = (test_spread_k1 - test_spread_elbow) / test_spread_k1 * 100

print()
print("Train rows:", len(train_members))
print("Test rows:", len(test_members))
print()
print(
    pd.DataFrame(
        {"k": ks, "train_spread": train_spreads, "test_spread": test_spreads}
    ).to_string(index=False)
)
print()
print("Train elbow:", train_elbow)
print("Test elbow:", test_elbow)
print("Elbows match:", elbows_match)
print("Test spread at k=1:", test_spread_k1)
print(f"Test spread at elbow k={train_elbow}:", test_spread_elbow)
print(f"Percent drop: {percent_drop:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(ks, train_spreads, marker="o")
axes[0].axvline(train_elbow, linestyle="--", color="gray", label=f"elbow k={train_elbow}")
axes[0].set_title("Train")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Sum of squared distances")
axes[0].set_xticks(ks)
axes[0].legend()

axes[1].plot(ks, test_spreads, marker="o")
axes[1].axvline(test_elbow, linestyle="--", color="gray", label=f"elbow k={test_elbow}")
axes[1].set_title("Test")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Sum of squared distances")
axes[1].set_xticks(ks)
axes[1].legend()

fig.tight_layout()
fig.savefig("elbow_train_vs_test.png")

v1 = KMeans(n_clusters=4, n_init=10, random_state=42)
v1.fit(X_train_scaled)

members = members.copy()
members["segment"] = v1.predict(scaler.transform(members[features]))

print()
print("v1 segments (k=4), all members:")
for segment, group in members.groupby("segment"):
    under_20 = (group["monthly_extras_spend"] < 20).mean()
    print()
    print(f"Segment {segment}")
    print("Members:", len(group))
    print(group[features].mean().round(1).to_string())
    print(f"Share spending less than $20: {under_20:.1%}")


def decide(member):
    """Scale one member with the train scaler and assign them with the k=4 model.

    Returns the segment and whether they get the $10 credit. Only segment 0 does.
    """
    row = pd.DataFrame([{feature: member[feature] for feature in features}])
    segment = int(v1.predict(scaler.transform(row))[0])
    return segment, segment == 0


new_member = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}
print()
print("New member:", decide(new_member))


def monthly_net(spend, gets_credit):
    """Net from the credit for one member. No credit is $0."""
    if not gets_credit:
        return 0
    if spend < 20:
        return 5
    return -10


baseline_net = sum(
    monthly_net(spend, True) for spend in members["monthly_extras_spend"]
)
v1_net = sum(
    monthly_net(spend, segment == 0)
    for spend, segment in zip(members["monthly_extras_spend"], members["segment"])
)

print()
print("Baseline monthly net:", baseline_net)
print("v1 monthly net:", v1_net)
print("Difference (v1 - baseline):", v1_net - baseline_net)

import json, collections

def load_roles(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    roles = []
    for item in data:
        if isinstance(item, dict) and "role" in item:
            roles.append(item["role"])
    return collections.Counter(roles)

rwfb = load_roles("rwfb_role_predictions.json")
cets = load_roles("cetras_role_predictions.json")

all_roles = sorted(set(rwfb) | set(cets))
print("Role\tRWFB\tCETraS-like")
for r in all_roles:
    print(f"{r}\t{rwfb.get(r,0)}\t{cets.get(r,0)}")

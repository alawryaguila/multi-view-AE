

def update_dict(orig_dict, update_dict):
    for key, val in update_dict.items():
        if key in orig_dict.keys():
            orig_dict[key] = val
    return orig_dict
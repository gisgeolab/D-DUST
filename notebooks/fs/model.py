def remove_int_values(labels):
    for l in labels:
        if('int' in l):
            labels.remove(l)
    return labels
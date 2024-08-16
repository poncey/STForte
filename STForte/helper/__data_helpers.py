import pickle


def load_gdata(path):
    with open(path, "rb") as f:
        gdata = pickle.load(f)
    return gdata


def save_gdata(gdata,
               path):
    with open(path, "wb") as f:
        pickle.dump(gdata, f)

import pickle
def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    list = load_obj("master_index")
    print("done")

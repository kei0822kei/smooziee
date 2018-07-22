from smooziee.smooziee.fitting import Fitting
import pickle


with open('../../notebook/pksearch.pickle', 'rb') as f:
    processor = pickle.load(f)

fitter = Fitting(['lorentzian' for i in range(len(processor.ix_peaks))])

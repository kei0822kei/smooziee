import re
import lmfit


"""
### CODING NOTICE ###

# Naming
function names ... Follow "lmfit.lineshapes" function
                    ('lorentzian', 'gaussian', ...)
parameter names ... Each function's parameters which are noticed
                    in lmfit's documentation of "built-in models"
                    (for lorentzian, 'amplitude', 'center', ...)

"""


epsilon = 1e-8


class Fitting():

    def _init__(self, peak_funcs):
        """
        Inputs
        ------
        peak_funcs: list of str
            ex. ['lorentzian', 'gaussian', ...]

        """
        self.model = sum(self._model(i, peak_func)
                         for i, peak_func in enumerate(peak_funcs))
        self.params = self.model.make_params()

    def _model(i, peak_func):
        if peak_func == 'lorentzian':
            prefix = 'l' + str(i) + '_'
            return lmfit.models.LorentzianModel(prefix=prefix)
        elif peak_func == 'gaussian':
            prefix = 'g' + str(i) + '_'
            return lmfit.models.GaussianModel(prefix=prefix)

    def set_params_expr(self, ix_peaks, ix_peakpairs, params):
        """
        Inputs
        ------
        ix_peaks: list of int
            ex. [36, 62, 97]
        ix_peakpairs: list of list
            ex. [[36, 97], [...], ...]
        params: list of str
            ex. ['amplitude', 'center']

        """
        def same_idx(i):
            same_idx = None
            for ix_peakpair in ix_peakpairs:
                if self.ix_peakpair[i] == pair_idx_lst[1]:
                    same_idx = self.peak_idx_lst.index(pair_idx_lst[0])
            return same_idx

        for i, param_name in enumerate(self.model.param_names):
            for param in params:
                self.params['g1_'+param].set(expr=param_name+'_'+str(same_idx(i)))

    def set_params_vary(self, i_peaks, params, vary):
        """
        i_peaks: list of int
            index of peaks to fix  ex.[2, 9]
        params: list of str
            ex. ['amplitude', 'center']
        vary: bool

        """
        def param_name(i_peak, param):
            match_names = [re.match('^[a-zA-Z]+%d_%s' % (i_peak, param),
                                    param_name)
                           for param_name in self.model.param_names]
            assert len(match_names) == 1
            return match_names[0]

        for i_peak in i_peaks:
            for param in params:
                self.params[param_name(i_peak, param)].set(vary=vary)

    def fit(self, x, y):
        self.model.fit(y, self.params, y)

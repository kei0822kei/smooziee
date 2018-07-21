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

    def __init__(self, peak_funcs):
        """
        Inputs
        ------
        peak_funcs: list of str
            ex. ['lorentzian', 'gaussian', ...]

        """
        models = [self._model(i, peak_func)
                  for i, peak_func in enumerate(peak_funcs)]
        self.model = sum(models[1:], models[0])
        self.params = self.model.make_params()

    def _model(self, i, peak_func):
        """
        Have to set in order that self._param_name() can convert

        """
        if peak_func == 'lorentzian':
            prefix = 'l' + str(i) + '_'
            return lmfit.models.LorentzianModel(prefix=prefix)
        elif peak_func == 'gaussian':
            prefix = 'g' + str(i) + '_'
            return lmfit.models.GaussianModel(prefix=prefix)

    def _param_name(self, i_peak, param):
        """
        ex. i_peak=1, param='sigma' -> 'g1_sigma'

        """
        r = re.compile('^[a-zA-Z]+%d_%s' % (i_peak, param))
        match_names = [param_name for param_name in self.model.param_names
                       if r.match(param_name)]
        if len(match_names) != 1:
            raise ValueError("'%s_%s' match %s" % (i_peak, param, match_names))
        return match_names[0]

    def set_params_expr(self, ix_peaks, ix_peakpairs, params):
        """
        Inputs
        ------
        ix_peaks: list of int (must be sorted)
            ex. [36, 62, 97]
        ix_peakpairs: list of list (must be sorted)
            ex. [[36, 97], [...], ...]
        params: list of str
            ex. ['amplitude', 'center']

        """
        def pair_i_peak(ix_peak):
            pair_i_peak = None
            for ix_peakpair in ix_peakpairs:
                if ix_peakpair[1] == ix_peak:
                    pair_i_peak = ix_peaks.index(ix_peakpair[0])
            return pair_i_peak

        for param in params:
            for i, ix_peak in enumerate(ix_peaks):
                if pair_i_peak(ix_peak) is not None:
                    self.params[self._param_name(i, param)].set(
                        expr=self._param_name(pair_i_peak(ix_peak), param))

    def set_params_vary(self, i_peaks, params, vary):
        """
        i_peaks: list of int
            index of peaks to fix  ex.[2, 9]
        params: list of str
            ex. ['amplitude', 'center']
        vary: bool

        """
        for i_peak in i_peaks:
            for param in params:
                self.params[self._param_name(i_peak, param)].set(vary=vary)

    def fit(self, x, y):
        self.model.fit(y, self.params, y)

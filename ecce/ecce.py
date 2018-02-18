# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as spd
from ..simimeasures import cy as cy
from ..simimeasures import py as py


class ECCE(object):

    def __init__(self, sim_func, inpt_vectz=1, agg_comb='product'):

        if sim_func == 'cosine_sim':
            self.sim_func = cy.cosine_sim
        elif sim_func == 'py_cosine_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.cosine_sim

        print "Initialing ECCE..."

        # Selecting the fit() and predict() methods wheather the...
        # ...number of input vectors are 1 or more.
        if inpt_vectz == 1:

            self.fit = self._fit
            self.predict = self._predict

        else:

            raise Expression("Mutiple input vectors implementation not ready yet.")

            # Initialing these object variables only when they are going to be used later.
            self.inpt_vectz = inpt_vectz
            self.agg_comb = agg_comb

            self.fit = self._fit_multi
            self.predict = self._predict_multi

        # RFSE paramters
        self.ci2gtag = dict()
        self.gnr_csums = list()
        self.gnr_ssums = list()
        self.grn_snum = list()

    def _fit(self, trn_mtrx, cls_tgs):

        # Preventing '0' class-tag usage as Known-class tag.
        if np.min(cls_tgs) == 0:
            msg = "Class tag '0' not allowed because 0 class indicates Uknown-Class " +\
                    "in the Open-set Classification framework"
            raise Exception(msg)

        # Calculating the initial sigma thresholds on Centroids.
        for i, gnr_tag in enumerate(np.unique(cls_tgs)):

            # Keeping the Genre's class tag.
            self.ci2gtag[i] = gnr_tag

            # Getting the Genre's training indecies.
            gnr_set_idxs = np.where((cls_tgs == gnr_tag))[0]

            # Creating the initial Class Centroids.
            # TO BE RELPACED WITH CYTHON
            sum_cvect = trn_mtrx[gnr_set_idxs].sum(axis=0)
            cls_cvect = (sum_cvect / np.linalg.norm(sum_cvect))
            cls_cvect = cls_cvect.reshape((1, cls_cvect.size))

            # Calculating the similarities of the Genre's traning set pages with the...
            # ...Genre's Centroind.
            trn_set_sims = np.array(self.sim_func(trn_mtrx, gnr_set_idxs, cls_cvect))
            trn_set_sims = trn_set_sims.reshape(trn_set_sims.shape[0])

            # Calculating the Similarity sums and Sigma threshold for this class centroid.
            # self.gnr_ssums.append(np.sum(trn_set_sims))
            gnr_simga = np.sum(trn_set_sims) / float(gnr_set_idxs.size)

            # Re-calculating the genres centroids by rejecting the outages, i.e. the pages...
            # ...where their similarity is greater than sigma threshold.
            # NOTE: DOUBLE CHECK THIS
            new_trn_set_idxs = gnr_set_idxs[np.where(trn_set_sims >= gnr_simga)]

            # Keeping the new traning set's amount of samples for this Genre Class.
            self.grn_snum.append(new_trn_set_idxs.size)

            # Adjusting the new Centroid for this Genre class.
            self.gnr_csums.append(trn_mtrx[new_trn_set_idxs].sum(axis=0))

            # Adjusting the new Sigma for this Gerne class.
            # NOTE: Should I do this step? In paper Does not!
            trn_set_sims = np.array(self.sim_func(trn_mtrx, new_trn_set_idxs, cls_cvect))
            trn_set_sims = trn_set_sims.reshape(trn_set_sims.shape[0])
            self.gnr_ssums.append(np.sum(trn_set_sims))

        # Converting the lists into narrays.
        self.gnr_csums = np.vstack(self.gnr_csums)
        self.gnr_ssums = np.hstack(self.gnr_ssums)
        self.grn_snum = np.array(self.grn_snum, dtype=np.float)

        return self.gnr_csums, self.gnr_ssums, self.grn_snum, self.ci2gtag

    def _fit_multi(self, *args):
        pass

    def _predict(self, tst_mtrx):

        # Initilising Predicted Scores and Predicted Y for the class tags to be returned
        self.predicted_scores = list()
        self.predicted_Y = list()

        # Getting test samples similaries by presenting them in random order to the learners...
        # ...of the ensemble. Then deciding the class tag given the best similary score over sigma.
        # NOTE: The Centroids and the Sigma are gradualy moving towards to the new notion of...
        # ...the genre.
        for smpl_i in np.random.permutation(tst_mtrx.shape[0]):

            # Normalizing the Sum-Centroid of all Genre Classes.
            grn_ctrds = self.gnr_csums / np.linalg.norm(self.gnr_csums)

            # Calculating the similarities of the random test pages with all the...
            # ...Genre Centroinds.
            tst_i_csims = np.array(self.sim_func(tst_mtrx, np.array([smpl_i]), grn_ctrds))[0]

            # Getting the max-similar centroid index.
            maxsim_ci = np.argmax(tst_i_csims)

            # Checking the max-similarity with the sigma-threshold for decising wheather to...
            # ...classify the sample to the most similar genre or to discart it as noise.
            if tst_i_csims[maxsim_ci] >= self.gnr_ssums[maxsim_ci] / self.grn_snum[maxsim_ci]:

                # Adjusting the max-similar centroid's new sum values. Adding the samples that is...
                # ...very close to this centroid for adjusting the centroid for the next sample.
                self.gnr_csums[maxsim_ci] = np.sum(
                    [self.gnr_csums[maxsim_ci], tst_mtrx[smpl_i]], axis=0
                )

                # Adjusting the sigma thresholds for this class with the new value.
                self.gnr_ssums[maxsim_ci] = np.sum(
                    [self.gnr_ssums[maxsim_ci], tst_i_csims[maxsim_ci]]
                )

                # Adding one to the amount of samples cosist for this Gerne class centroid's...
                # ...caclulation.
                self.grn_snum[maxsim_ci] += 1
                self.predicted_scores.append(tst_i_csims[maxsim_ci])
                self.predicted_Y.append(self.ci2gtag[maxsim_ci])

            else:

                self.predicted_scores.append(0.0)
                self.predicted_Y.append(0)

        # Converting the results list to numpy arrays.
        self.predicted_scores = np.array(self.predicted_scores, dtype=np.float)
        self.predicted_Y = np.array(self.predicted_Y, dtype=np.int)

        # NOTE: returning the adjusted Centroid, Similarity Sums and Samples amount per class...
        # ...after the adjustments happend during the prediction phase.
        return self.predicted_Y, self.predicted_scores,\
            self.gnr_csums, self.gnr_ssums, self.grn_snum

    def _predict_multi(self, tst_mtrx):
        pass

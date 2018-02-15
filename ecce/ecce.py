# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as spd
from ..simimeasures import cy as cy
from ..simimeasures import py as py


class ECCE(object):

    def __init__(self, sim_func, inpt_vectz=1, agg_comb='product'):

        if sim_func == 'cosine_sim':
            self.sim_func = cy.cosine_sim
            self.sim_min_value = -1.0
        elif sim_func == 'eucl_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = cy.eucl_sim
            self.sim_min_value = 0.0
        elif sim_func == 'minmax_sim':
            self.sim_func = cy.minmax_sim
            self.sim_min_value = 0.0
        elif sim_func == 'py_cosine_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.cosine_sim
            self.sim_min_value = -1.0
        elif sim_func == 'py_cos_sim_sprs':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.cos_sim_sprs
            self.sim_min_value = -1.0
        elif sim_func == 'py_minmax_sim':
            # print "NOT TESTED - NOT WORKING"
            self.sim_func = py.minmax_sim
            self.sim_min_value = 0.0
        elif sim_func == 'py_jaccard_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.jaccard_sim
            self.sim_min_value = 0.0
        elif sim_func == 'py_hamming_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.hamming_sim
            self.sim_min_value = 0.0
        elif simfunc == 'py_correl_sim':
            print "NOT TESTED - NOT WORKING"
            self.sim_func = py.correl_sim
            self.sim_min_value = 0.0

        print "Initialing ECCE..."

        # Selecting the fit() method whether the number of input vectors are 1 or more.
        if inpt_vectz == 1:

            fit = _fit

        else:

            raise Expression("Mutiple input vectors implementation not ready yet.")

            # Initialing these object variables only when they are going to be used later.
            self.inpt_vectz = inpt_vectz
            self.agg_comb = agg_comb

            fit = _fit_multi

        # RFSE paramters
        self.ci2gtag = dict()
        self.gnr_classes = list()
        self.gnr_sigma = list()

    def _fit(self, trn_mtrx, cls_tgs):

        # Preventing '0' class-tag usage as Known-class tag.
        if np.min(cls_tgs) == 0:
            msg = "Class tag '0' not allowed because 0 class indicates Uknown-Class " +\
                    "in the Open-set Classification framework"
            raise Exception(msg)

        # Creating the initial Class Centroids.
        for i, gnr_tag in enumerate(np.unique(cls_tgs)):
            self.ci2gtag[i] = gnr_tag
            self.gnr_classes.append(trn_mtrx[np.where((cls_tgs == gnr_tag))].mean(axis=0))

        # Calculating the initial sigma thresholds on Centroids.
        for i, gnr_tag in enumerate(np.unique(cls_tgs)):

            # Keeping the Genre's class tag.
            self.ci2gtag[i] = gnr_tag

            # Getting the Genre's training indecies.
            gnr_set_idxs = np.where((cls_tgs == gnr_tag))

            # Creating the initial Class Centroids.
            self.gnr_classes.append(trn_mtrx[gnr_set_idxs].mean(axis=0))

            # Calculating the similarities of the Genre's traning set pages with the...
            # ...Genre's Centroind.
            trn_set_sims = np.array(self.sim_func(trn_mtrx, gnr_set_idxs, self.gnr_classes))

            # Calculating the Sigma thresholds.
            self.gnr_sigma.append(np.sum(trn_set_sims) / float(gnr_set_idxs.size))

            # Re-calculating the genres centroids by rejecting the outages, i.e. the pages...
            # ...where their similarity is greater than sigma threshold.
            # NOTE: DOUBLE CHECK THIS
            new_trn_set_idxs = gnr_set_idxs[np.where(trn_set_sims <= self.gnr_sigma[i])]

            # Calculating the new Centroid for this Genre class.
            self.gnr_classes[i] = trn_mtrx[new_trn_set_idxs].mean(axis=0)

        # Converting the lists into narrays.
        self.gnr_classes = np.vstack(self.gnr_classes)
        self.gnr_sigma = np.hstack(self.gnr_sigma)

        return self.gnr_classes, self.gnr_sigma, self.ci2gtag[i]

    def _fit_multi(self, *args):
        pass

    def _predict(self, tst_mtrx):
        # It should be tst_mtrx = corpus_mtrx[crv_idxs]
        # It should be cls_tgs = cls_gnr_tgs[crv_idxs]

        mtrx_feat_idxs = np.arange(tst_mtrx.shape[1])

        max_sim_scores_per_iter = np.zeros((self.itrs, tst_mtrx.shape[0]))
        predicted_classes_per_iter = np.zeros((self.itrs, tst_mtrx.shape[0]), dtype=np.int)

        # Measure similarity for i iterations i.e. for i different feature subspaces Randomly...
        # ...selected
        for i in np.arange(self.itrs):

            # Construct Genres Class Vectors form Training Set. In case self.bagging is True.
            if self.bagging:
                self.gnr_classes = self.contruct_classes(self.trn_mtrx, self.cls_tgs)

            # Randomly select some of the available features
            feat_subspace = np.random.permutation(mtrx_feat_idxs)[0:self.feat_size]

            # Initialized Predicted Classes and Maximum Similarity Scores Array for this i iteration
            sim_scrs = np.array(self.sim_func(tst_mtrx, self.gnr_classes, feat_subspace))
            max_sim_inds = np.argmax(sim_scrs, axis=1)
            max_sim_scores = sim_scrs[np.arange(sim_scrs.shape[0]), max_sim_inds]

            # Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[i, :] = max_sim_scores[:]
            predicted_classes_per_iter[i, :] = np.array([self.ci2gtag[j] for j in max_sim_inds[:]])

        # Getting the Max Score and the respective prediction where the score is gte to the...
        # ...sigma threshold. If lte than threshold then prediction is 0, i.e 'uknown class'...
        # ...and the score is set to 0.
        classes_num = np.max(self.ci2gtag.values()) + 1  # Appeding 1 for '0' class-tag counting.
        genres_occs = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=classes_num), axis=0,
            arr=predicted_classes_per_iter.astype(np.int)
        )
        genres_probs = genres_occs / np.float(self.itrs)

        # Getting the scores over sigma, and setting the rest to 0.0.
        scors_over_sigma = np.where(genres_probs > self.sigma, genres_probs, 0.0)

        # Getting the Max Score and the respective predicted_Y over simga threshold.
        predicted_Y = np.argmax(scors_over_sigma, axis=0)
        predicted_scores = np.max(scors_over_sigma, axis=0)

        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter

    def _predict_multi(self, tst_mtrx):
        pass
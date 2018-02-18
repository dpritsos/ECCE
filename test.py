
import sys
import numpy as np

sys.path.append('../')
import ECCE as ecce

if __name__ == "__main__":

    trn_mtrx = np.vstack(
        (
            0.1 * np.random.randn(5, 3) + 5,
            0.1 * np.random.randn(5, 3) + 5,
            0.1 * np.random.randn(5, 3) + 0.5,

            0.1 * np.random.randn(5, 3) + 10,
            0.1 * np.random.randn(5, 3) + 10,
            0.2 * np.random.randn(5, 3) + 0.6,

            0.2 * np.random.randn(5, 3) + 100,
            0.1 * np.random.randn(5, 3) + 100,
            0.1 * np.random.randn(5, 3) + 0.6,
        )
    )
    tst_mtrx = np.vstack(
        (
            0.1 * np.random.randn(3, 3) + 5,
            0.1 * np.random.randn(3, 3) + 10,
            0.1 * np.random.randn(3, 3) + 100,
            0.1 * np.random.randn(3, 3) + 1000,
            0.1 * np.random.randn(3, 3) + 0.5,

        )
    )
    cls_tags = np.array(10*[1] + 10*[2] + 10*[3])
    # print trn_mtrx

    ecce_mdl = ecce.ECCE(sim_func='cosine_sim', inpt_vectz=1, agg_comb='product')

    ecce_mdl.fit(trn_mtrx, cls_tags)

    #print trn_mtrx
    #print tst_mtrx

    rs = ecce_mdl.predict(tst_mtrx)
    print rs[0]
    print rs[1]

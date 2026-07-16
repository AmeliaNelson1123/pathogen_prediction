"""Train and persist the deployed models from the CV-selected best configs.

Single source of truth: reads preparation/data_results/best_configs.json so the
served models match the reported metrics. Run AFTER Run_and_Test_Models.ipynb.
"""
from pathlib import Path
import joblib, numpy as np
import preparation.pipeline_utils as pu

OUTPUT = pu.project_root() / "website" / "backend" / "models"
OUTPUT.mkdir(parents=True, exist_ok=True)


def _feature_frame(df, variant):
    X = df.drop(columns=[pu.Y_COL])
    if variant == "longlat_only":
        keep = [c for c in pu.LONGLAT_VARS_ONLY if c in X.columns]
        X = X[keep]
    elif variant == "soil_only":
        keep = [c for c in pu.SOIL_VARS_ONLY if c in X.columns]
        X = X[keep]
    out = X.copy(); out[pu.Y_COL] = df[pu.Y_COL].values
    return out


def _clf_params(cfg):
    return {k.replace("clf__", ""): v for k, v in cfg["params"].items()}


def train_variant(df, variant, configs):
    pu.set_seeds()
    sub = _feature_frame(df, variant)
    Xtr, _, ytr, _ = pu.make_train_test(sub)

    pre = pu.make_preprocessor(add_clusters=(variant == "main"))
    Xtr_t = pre.fit_transform(Xtr)
    joblib.dump(pre, OUTPUT / f"preprocess_{variant}.joblib")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    gbm = GradientBoostingClassifier(random_state=pu.RANDOM_STATE, **_clf_params(configs["gbm"]))
    gbm.fit(Xtr_t, ytr); joblib.dump(gbm, OUTPUT / f"gbm_{variant}.joblib")

    svm = SVC(probability=True, max_iter=20000, random_state=pu.RANDOM_STATE, **_clf_params(configs["svm"]))
    svm.fit(Xtr_t, ytr); joblib.dump(svm, OUTPUT / f"svm_{variant}.joblib")

    nn_p = configs["neural_net"]["params"]
    nn = pu.build_nn(Xtr_t.shape[1], nn_p["n_layers"], nn_p["n_neurons"])
    nn.fit(Xtr_t, np.asarray(ytr), epochs=nn_p["epochs"], batch_size=nn_p["batch_size"], verbose=0)
    nn.save(OUTPUT / f"neural_net_{variant}.keras")
    print(f"saved {variant}: gbm, svm, neural_net, preprocess")


def main():
    df = pu.load_and_prep()
    configs = pu.load_best_configs()
    for variant in ["main", "longlat_only", "soil_only"]:
        train_variant(df, variant, configs)


if __name__ == "__main__":
    main()

import uuid
import json
from collections import defaultdict
from functools import reduce

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs
from fdllm import get_caller
from fdllm.extensions import general_query, ADict

def auto_tag_generate(
    cluster_labs,
    cluster_scores,
    topnchunks,
    chunk_size,
    caller="gpt-4-1106-preview",
    n_candidates=3,
):

    cluster_descriptions, unqtags = tag_clusters(
        topnchunks,
        n_candidates=n_candidates,
        first_pass_only=False,
        caller=caller,
    )
    ######
    tags = []
    tagtype = "general_topic"
    for cd in cluster_descriptions:
        tags.append([tg["tag"] for tg in cd["tags"][tagtype]])
    #
    flattags = reduce(lambda a, b: set(a).union(set(b)), tags)
    #####

    return tags, flattags


def tag_clusters(
    topnchunks: dict,
    caller=None,
    n_candidates=4,
    first_pass_only=False,
    verbose=0,
):
    if caller is None:
        caller = get_caller("gpt-4-1106-preview")
    elif isinstance(caller, str):
        caller = get_caller(caller)

    n_top = len(next(iter(topnchunks.values())))

    DESCR = {
        "id": None,
        "purpose:: what is this text trying to do? (list of tags)": [None],
        "general_topic:: (list of tags)": [None],
        "specific_topic:: (list of tags)": [None],
    }

    giveup = n_candidates + 4
    out = []
    cnt = 0
    while len(out) < n_candidates:
        if cnt > giveup:
            raise ValueError()

        jsonin = ADict(
            {
                "samples"
                f":: {n_top} samples of text from documents "
                f"taken from {len(topnchunks)} different categories": [
                    {"id:: category id code": key, "samples": val}
                    for key, val in topnchunks.items()
                ]
            }
        )

        jsonout = ADict(
            {
                "descriptors"
                ":: How would you describe each document category?"
                " Differentiate each category from the others"
                " (i.e. avoid tags that are shared by every category)."
                " Return exactly one descriptor for each category.": [DESCR]
            }
        )

        out_ = general_query(jsonin, jsonout, caller)
        try:
            if set(dscr["id"] for dscr in out_["descriptors"]) == set(topnchunks):
                out.append(out_)
        except:
            pass
            cnt += 1

    if verbose > 0:
        print(json.dumps(out, indent=4))
        print([len(out_["descriptors"]) for out_ in out])

    ########
    unqtags = defaultdict(set)
    for out_ in out:
        for out__ in out_["descriptors"]:
            for key, val in out__.items():
                if isinstance(val, list):
                    unqtags[key].update(val)

    unqtags = {key: list(val) for key, val in unqtags.items()}

    if first_pass_only:
        return unqtags
    ########

    tags = []
    for cluster_id, samples in topnchunks.items():
        jsonin = ADict(
            {
                "samples"
                f":: {n_top} samples of text from documents": [
                    {"id:: category id code": cluster_id, "samples": samples}
                ]
            }
        )

        jsonout = ADict(
            {
                "descriptors"
                ":: How would you describe the documents in this category?": {
                    "tags:: select all that apply with a score greater than 5": {
                        f"{key}::"
                        f"[{'/'.join(val)}]": [
                            {
                                "tag": None,
                                "score:: 1 (low relevance) to 10 (high relevance)": None,
                            }
                        ]
                        for key, val in unqtags.items()
                    },
                }
            }
        )
        cnt = 0
        while cnt < giveup:
            try:
                out = general_query(jsonin, jsonout, caller)
            except:
                cnt += 1
                continue
            tags_ = out["descriptors"]["tags"]
            break
        if cnt == giveup:
            raise ValueError()
        tags.append({"id": cluster_id, "tags": tags_})

    # jsonin = ADict(
    #     {
    #         "samples"
    #         f":: {n_top} samples of text from documents "
    #         f"taken from {len(topnchunks)} different categories": [
    #             {"id:: category id code": key, "samples": val}
    #             for key, val in topnchunks.items()
    #         ]
    #     }
    # )

    # jsonout = ADict(
    #     {
    #         "descriptors"
    #         ":: How would you describe each document category?": [
    #             {
    #                 "id": None,
    #                 "tags:: select all tags that apply from each section": {
    #                     f"{key}::" f"[{'/'.join(val)}]": []
    #                     for key, val in unqtags.items()
    #                     # if key == "general_topic"
    #                 },
    #             }
    #         ]
    #     }
    # )

    # cnt = 0
    # while cnt < giveup:
    #     try:
    #         out = general_query(jsonin, jsonout, caller)
    #     except:
    #         cnt += 1
    #         continue
    #     tags = out["descriptors"]
    #     if set(tgs["id"] for tgs in tags) == set(topnchunks):
    #         break
    # if cnt == giveup:
    #     raise ValueError()

    out = []
    for cluster_id in topnchunks:
        gottags = [tgs for tgs in tags if tgs["id"] == cluster_id]
        if len(gottags) != 1:
            raise ValueError()
        out.append(gottags[0])
    return out, unqtags

    # jsoninnew = {
    #     "candidate_descriptors"
    #     f":: {n_candidates} descriptors for each category.": [
    #         {"candidate": i, "descriptors": ov["descriptors"]}
    #         for i, ov in enumerate(out)
    #     ]
    # }
    # jsoninnew = ADict({**jsonin, **jsoninnew})

    # jsonout = ADict(
    #     {
    #         "descriptors"
    #         f":: merge the {n_candidates} candidates into a single set of descriptors"
    #         " - one for each category."
    #         #" Rare and redundant tags can both be dropped.": [DESCR]
    #         : [DESCR]
    #     }
    # )

    # newout = general_query(jsoninnew, jsonout, caller)
    # cnt = 0
    # while set(dscr["id"] for dscr in newout["descriptors"]) != set(topnchunks):
    #     if cnt > giveup:
    #         raise ValueError()
    #     newout = general_query(jsoninnew, jsonout, caller)
    #     cnt += 1

    # if verbose > 0:
    #     print(json.dumps(newout, indent=4))
    #     print(len(newout["descriptors"]))

    # out = newout["descriptors"]
    # for dr in out:
    #     dr["id"] = list(topnchunks).index(dr["id"])

    # return out


def cluster_score(
    X,
    nc_range=np.arange(3, 20),
    clusterer=KMeans,
    n_fold=5,
    nreps=10,
    random_state=None,
):
    if nreps > 1:
        seeds = np.random.default_rng(random_state).integers(
            low=0, high=2**32 - 1, size=nreps
        )
        scores = [
            cluster_score(
                X=X, nc_range=nc_range, n_fold=n_fold, nreps=1, random_state=seeds[i]
            )
            for i in tqdm(range(nreps))
        ]
        return np.concatenate(scores[None], axis=0)

    kf = KFold(n_splits=n_fold)
    score = np.zeros((n_fold, len(nc_range)))
    for i, (train, test) in enumerate(kf.split(X)):
        for j, nc in enumerate(nc_range):
            cl = clusterer(n_clusters=nc, random_state=random_state, init="random")
            cl = cl.fit(X[train])
            pred = cl.predict(X[test])
            score[i, j] = silhouette_score(X[test], pred)
    return score


def cluster_silhouette_analysis(
    X,
    plot_vectors,
    colmap=cm.nipy_spectral,
    nc_range=np.arange(3, 7),
    clusterer=KMeans,
):
    fig, allaxs = plt.subplots(nrows=len(nc_range), ncols=2, figsize=(10, 8))
    fig.tight_layout(h_pad=2)
    for nc, axs in zip(nc_range, allaxs):
        axs[0].set_title(f"N Clusters: {nc}")
        axs[0].set_ylim([0, X.shape[0] + (nc + 1) * 10])
        cl = clusterer(n_clusters=nc, random_state=512782, init="random")
        labs = cl.fit_predict(X)
        sscore = silhouette_score(X, labs)
        axs[0].axvline(x=sscore, color="red", linestyle="--")
        ssamples = silhouette_samples(X, labs)
        ylower = 10
        subsscore = {"NC": nc, "Total": sscore}
        for c in range(nc):
            ssamples_ = np.sort(ssamples[labs == c])
            subsscore[c] = np.percentile(ssamples_, 99) > sscore
            ssize = ssamples_.shape[0]
            y_upper = ssize + ylower
            color = colmap(float(c) / nc)
            axs[0].fill_betweenx(
                np.arange(ylower, y_upper),
                0,
                ssamples_,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ylower = y_upper + 10
        print(subsscore)
        axs[1].scatter(*plot_vectors.T, s=10, c=colmap(labs.astype(float) / nc))


def cluster_simdata(n_clusters=6, cluster_std=1, random_state=456, gamma=1.0):
    rng = np.random.default_rng(random_state)
    n_clusters = 6
    rel_samples = rng.dirichlet(rng.uniform(size=n_clusters) ** gamma)
    n_samples = np.round(5000 * rel_samples).astype(int)
    simX, simlabs = make_blobs(
        n_samples=n_samples,
        n_features=2,
        cluster_std=cluster_std,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=random_state,
    )
    return simX, simlabs

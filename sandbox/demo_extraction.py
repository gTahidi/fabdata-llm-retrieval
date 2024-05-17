# %%
#! %matplotlib qt

from pathlib import Path

from fdllm.sysutils import register_models
from fdllmret.helpers.encoding import DocsetEncoding

ROOT = Path(__file__).resolve().parents[1]

# %%
docenc = DocsetEncoding.from_config(ROOT / "data_config.yml")

# %%
docenc.docembs.tsne.plot("all")
docenc.docembs.tsne.plot("all", doc_agg=True)

# # %%
# cs = 800
# docenc.auto_tag(cs)

# %%

# # %%
# unqtagembs = {key: get_embeddings(val) for key, val in unqtags.items()}

# # %%
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# alltagembs = np.concatenate(list(unqtagembs.values()), axis=0)
# chunkembs = docenc.docembs.matrix(cs)
# allcombembs = np.concatenate([alltagembs, chunkembs], axis=0)

# tsne = TSNE(random_state=62356235)

# vectors = tsne.fit_transform(allcombembs)

# labs = np.concatenate([np.ones(len(alltagembs)) * 0, np.ones(len(chunkembs)) * 1])

# fix, ax = plt.subplots()
# ax.scatter(*vectors.T, c=labs)

# # %%
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy import stats

# clembs = docenc.docembs.matrix(cs)[cluster_labs == 4]

# toptags = {}
# for key, embs in unqtagembs.items():
#     cc = np.corrcoef(np.concatenate([clembs, unqtagembs[key]]))[
#         : clembs.shape[0], clembs.shape[0] :
#     ]
#     cosim = cosine_similarity(clembs, unqtagembs[key])
#     topidx = stats.mode(np.argsort(cosim)[:, ::-1][:, :3], axis=0).mode
#     toptags[key] = [tg for i, tg in enumerate(unqtags[key]) if i in topidx]

# toptags


# # %%
# jsondata = copy.deepcopy(docenc.jsondata)


# # %%
# topnchunks = get_topnchunks(docenc, cosim, 3, cluster_labs, scores, n_start=0)

# print(json.dumps(topnchunks, indent=4))

# topnchunks = get_topnchunks(
#     docenc, cs, 5, cluster_labs, scores, n_start=6, id_strings=list(topnchunks)
# )

# print(topnchunks)


# %%

# %%
# cs = 800
# X = docset.docembs.matrix(cs)
# nc_range = np.arange(3, 12)
# score = cluster_score(X, nc_range=nc_range, n_fold=2)
# sm = score.mean(axis=0)

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.plot(nc_range, sm)

# nc_range[np.argmax(np.abs((sm[:-1] - sm[1:])) / np.abs(sm).max())]


# bgm = BayesianGaussianMixture(
#     n_components=20,
#     random_state=72632532,
#     # weight_concentration_prior=0.3,
#     init_params="k-means++",
#     # n_init=1,
# )
# labs = bgm.fit_predict(embeddings.allembmat)

# fig, ax = plt.subplots(figsize=(8, 8))
# fig.tight_layout()
# ax.scatter(*allvectors.T, s=6, c=labs / labs.max())

# # %%
# bgm = BayesianGaussianMixture(
#     n_components=20,
#     random_state=72632532,
#     # weight_concentration_prior=0.3,
#     # init_params="k-means++",
#     # n_init=1,
# )
# labs = bgm.fit_predict(allvectors)

# fig, ax = plt.subplots(figsize=(8, 8))
# fig.tight_layout()
# ax.scatter(*allvectors.T, s=6, c=labs / labs.max())

# # %%
# cs = 1000
# emb = docembmats[cs]
# tsnevecs = doc_tsne_vecs[cs]
# bgm = BayesianGaussianMixture(
#     n_components=15,
#     random_state=72632532,
#     # weight_concentration_prior=1e-10,
#     # weight_concentration_prior_type="dirichlet_distribution",
#     init_params="k-means++",
#     # n_init=5,
# )
# labs = bgm.fit_predict(emb)

# fig, ax = plt.subplots(figsize=(8, 8))
# fig.tight_layout()
# ax.scatter(*tsnevecs.T, s=10, c=labs / labs.max())

# # %%
# doclabs = {
#     id: {"filename": jsd["filename"], "labels": np.unique(labs[idx]).tolist()}
#     for (id, idx), jsd in zip(docidx[cs].items(), jsondata)
# }

# print(json.dumps(doclabs, indent=4))

# cs = 1000
# X = docset.docembs.matrix(cs)
# tsnevecs = docset.docembs.tsne(cs)

# km = KMeans(n_clusters=10, random_state=5214242)
# labs = km.fit_predict(X)
# scores = silhouette_samples(X, labs)

# fig, ax = plt.subplots(figsize=(8, 8))
# fig.tight_layout()
# sct = ax.scatter(*tsnevecs.T, s=10, c=labs / labs.max())
# leg = ax.legend(
#     sct.legend_elements()[0], np.unique(labs), loc="lower left", title="Classes"
# )
# ax.add_artist(leg)

# doclabs = {
#     id: {
#         "filename": Path(jsd["filename"]).name,
#         "labels": np.unique(labs[idx]).tolist(),
#     }
#     for (id, idx), jsd in zip(docset.docembs.docidx[cs].items(), docset.jsondata)
#     # for (id, idx), jsd in zip(zip(docidx[cs], range(len(docidx[cs]))), jsondata)
# }

# print(json.dumps(doclabs, indent=4))

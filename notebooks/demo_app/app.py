import os
import pickle
import json

from scipy.spatial.distance import cdist
from utils import get_regex_expression, preprocess_data
from models import SearchResponseModel, PingResponseModel
from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI()

prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").strip("/")

class SearchService(object):
    regex = get_regex_expression()
    fse_model = None
    idx_slug = None
    bigrams_model = None

    @classmethod
    def get_model(cls):
        if cls.fse_model is None:
            cls.fse_model = pickle.load(open("model_storage/fse_model.pickle", "rb"))
        return cls.fse_model

    @classmethod
    def get_slugs(cls):
        if cls.idx_slug is None:
            cls.idx_slug = json.load(open("data/idx_to_slug.json", "r"))
        return cls.idx_slug

    @classmethod
    def get_bigrams(cls):
        if cls.bigrams_model is None:
            cls.bigrams_model = pickle.load(open("model_storage/bigrams.pickle", "rb"))
        return cls.bigrams_model

    @classmethod
    def get_artifacts(cls):
        return cls.get_slugs(), cls.get_model(), cls.get_bigrams()

    @classmethod
    def get_search_results(cls, s):
        slugs, model, bigrams = cls.get_artifacts()
        s_preprocessed = preprocess_data(s, cls.regex, True, True)
        s_ = list(bigrams[s_preprocessed])
        s_vector = model.infer([(s_, 0)])
        results = cdist(s_vector, model.sv.vectors, metric='cosine').squeeze()
        results = list(enumerate(results))
        results = sorted(results, key=lambda x: x[1])[:10]
        results = [slugs[idx] for idx, _ in results]
        return results        

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(prefix + "/docs")

@app.get("/ping", tags=["Endpoint Health"], response_model=PingResponseModel)
def ping():
    health = [artifact is not None for artifact in SearchService.get_artifacts()]
    status = 'ok' if len(health) == sum(health) else 'error'
    return {
        "status": f"{status}"
    }

@app.get("/search", tags=["Search Endpoint"], response_model=SearchResponseModel)
def search(s: str):
    results = SearchService.get_search_results(s)
    return {
        "results": results
    }
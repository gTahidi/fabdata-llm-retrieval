from pathlib import Path
from collections import defaultdict
import uuid
import json
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import warnings
from itertools import chain
from fnmatch import fnmatch
import requests
from io import StringIO
from types import SimpleNamespace
import copy
import multiprocessing as mp
import platform

from tqdm import tqdm
from pypdf import PdfReader
import fitz
from joblib import Parallel, delayed
from fdllm import get_caller
from fdllm.extensions import general_query
from fdllm.sysutils import register_models


def process_folder(
    folder,
    n_jobs=1,
    extraction_model="gpt-4-1106-preview",
    extract_refs=False,
    download_refs=False,
    custom_models_config=None,
    verbose=10,
    pdfengine="pypdf",
):
    if not isinstance(folder, list):
        folder = [folder]
    if custom_models_config is not None:
        cmcfile = process_path(custom_models_config)
        if not cmcfile.exists():
            raise IOError(f"Can''t find {custom_models_config}")
        else:
            register_models(cmcfile)

    if verbose > 0:
        print("Extracting text from documents")
    jsondata = extract_text_folders(
        folders=folder, n_jobs=n_jobs, verbose=verbose, pdfengine=pdfengine
    )
    if verbose > 0:
        print("Creating contents")
    contents = create_contents(folders=folder)
    if extract_refs:
        if verbose > 0:
            print("Extracting references from text")
        extract_references(
            jsondata, in_place=True, model=extraction_model, verbose=verbose
        )
    if download_refs:
        download_references(jsondata, in_place=True, verbose=verbose)

    return jsondata, contents


def process_path(path):
    path = Path(path)
    pathparts = [Path.home() if part == "~" else part for part in path.parts]
    return Path("/".join(pathparts)).resolve()


def filesgen(path, exts):
    path = Path(path)
    tags = _load_tags(path)
    for fl in path.rglob("*"):
        if fl.suffix in exts:
            relfl = fl.relative_to(path)
            yield fl, relfl.parent, _check_tags(relfl, tags)


def _check_tags(fl, tags):
    return [tag for tag, pats in tags.items() if any(fnmatch(fl, pat) for pat in pats)]


def _load_tags(path):
    tagsfile = path / "tags.json"
    if tagsfile.exists():
        with open(tagsfile) as f:
            tags = json.load(f)
        flattags = defaultdict(list)
        for tagset in tags:
            if tagset["key_type"] == "tag":
                for tag, pats in tagset["tags"].items():
                    flattags[tag].extend(pats)
            elif tagset["key_type"] == "pat":
                for pat, tags in tagset["tags"].items():
                    for tag in tags:
                        flattags[tag].append(pat)
        return flattags
    else:
        return {}


def _extract_text_pdf(file, engine="pypdf"):
    if isinstance(file, str):
        with StringIO(file) as f:
            return _extract_text_pdf(f)
    else:
        if hasattr(file, "name"):
            name = file.name
        else:
            name = None
        if engine == "pypdf":
            text = "\n".join(page.extract_text() for page in PdfReader(file).pages)
        elif engine == "fitz":
            with fitz.open(file) as pdf:
                text = "\n".join(page.get_text() for page in pdf.pages())
        else:
            raise NotImplementedError("Invalid reader")
        return name, text


def extract_text(file, exts, parent, tags, pdfengine="pypdf"):
    try:
        if file.suffix in [".pdf"]:
            name, text = _extract_text_pdf(file, pdfengine)
        elif file.suffix == ".zip":
            with ZipFile(file, mode="r") as zf, TemporaryDirectory() as td:
                zf.extractall(td)
                return [
                    extract_text(fl, exts, parent_)
                    for fl, parent_ in filesgen(td, exts)
                ]
        else:
            warnings.warn(f"{file} not supported filetype")
        return name, text, file.suffix, parent, tags
    except Exception as err:
        return file.name, None, file.suffix, parent, tags


def _flattener(pages, allflatpages=[]):
    flatpages = []
    for p in pages:
        if isinstance(p, list):
            flatpages.extend(_flattener(p, allflatpages=flatpages))
        else:
            if p not in flatpages and p not in allflatpages:
                flatpages.append(p)
    return flatpages


def extract_text_folders(
    folders, n_jobs=1, exts=[".pdf"], pdfengine="pypdf", verbose=0
):
    if n_jobs == -1 and platform.system() == "Windows":
        n_jobs = min(mp.cpu_count(), 61)
    datapath = [process_path(fold) for fold in folders]

    fgen = list(chain(*[filesgen(dp, exts) for dp in datapath]))

    batch_size = "auto"
    if n_jobs == 0:
        raise ValueError("njobs can't be 0")
    elif n_jobs != 1:
        p = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=batch_size)
        pages = p(
            delayed(extract_text)(file, exts, parent, tags, pdfengine)
            for file, parent, tags in fgen
        )
    else:
        if verbose > 0:
            flist = tqdm(list(fgen))
        else:
            flist = fgen
        pages = (
            extract_text(file, exts, parent, tags, pdfengine)
            for file, parent, tags in flist
        )

    flatpages = _flattener(pages)
    jsondata = []
    for key, pagestext, suffix, parent, tags in flatpages:
        name = name = (parent / key).as_posix()
        jsondata.append(
            {
                "id": str(uuid.uuid4()),
                "text": pagestext,
                "source": "file",
                "filename": name,
                "tag": ",".join(tags),
            }
        )
    # insert urls
    urlfiles = chain(*[((dp, fl) for fl in dp.rglob("*urls.json")) for dp in datapath])
    for dp, urlf in urlfiles:
        with open(urlf) as f:
            urls = json.load(f)
        base = urlf.relative_to(dp).parent.as_posix()
        for filedata in jsondata:
            if Path(filedata["filename"]).parent.as_posix() == base:
                for fname, url in urls.items():
                    key = (Path(base) / fname).as_posix()
                    if filedata["filename"] == key:

                        filedata["url"] = url

    return jsondata


def create_contents(folders):
    datapath = [Path(fold).resolve() for fold in folders]
    introfiles = chain(*[dp.rglob("*intro.txt") for dp in datapath])
    contents = {}
    for introf in introfiles:
        with open(introf) as f:
            contents[introf.parent.name] = f.read()

    return contents


def extract_references(jsondata, in_place=True, model="gpt-4-1106-preview", verbose=0):
    maxref = 30
    caller = get_caller(model)
    refs = []
    if verbose > 0:
        fileiter = tqdm(jsondata)
    else:
        fileiter = jsondata
    for file in fileiter:
        gotrefs = []
        # i = 0
        while True:
            # print(f"{file['filename']}: {i :03d}")
            # i += 1
            jsonin = {"document": file["text"][-60000:], "got_references": gotrefs}
            jsonout = {
                "references::"
                " References not already included in got_references."
                " Start counting on the first reference not already"
                " in got_references and stop listing references after you "
                f" reach the count of {maxref}.": [
                    {
                        "count:: (int)": None,
                        "authors": [],
                        "title": None,
                        "journal": None,
                        "volume": None,
                        "url": None,
                        "year": None,
                    }
                ]
            }
            resp = general_query(jsonin, jsonout, caller=caller)
            if not resp["references"]:
                break
            # print(json.dumps(resp, indent=4))
            gotrefs.extend(resp["references"])
            gotrefs_ = []
            for gr in gotrefs:
                gr = {k: v for k, v in gr.items() if k != "count"}
                if not any(gr_ == gr for gr_ in gotrefs_):
                    gotrefs_.append(gr)
            gotrefs = sorted(
                gotrefs_,
                key=lambda x: (
                    (True, auth[0])
                    if (auth := x["authors"]) is not None
                    else (False, auth)
                ),
            )
            # with open(outfile, "w") as f:
            #     json.dump(refs + gotrefs, f, indent=4)
            if len(resp["references"]) < maxref:
                break
            # if max(ref["count"] for ref in gotrefs) < maxref * i:
            #     break
        refs.append(gotrefs)

    if in_place:
        for ref, file in zip(refs, jsondata):
            file["refs"] = ref
    else:
        return refs


def download_references(jsondata, in_place=True, verbose=0):
    refdocs = defaultdict(lambda: defaultdict(dict))
    if verbose > 0:
        fileiter = tqdm(jsondata)
    else:
        fileiter = jsondata
    for file in fileiter:
        for ref in file.get("refs", []):
            if ref.get("url") is not None:
                print(f'URL: {ref["url"]}')
                try:
                    r = requests.get(ref["url"])
                except:
                    continue
                if r.status_code == 200:
                    try:
                        _, text = extract_text(r.content, suffix=".pdf")
                        doc = SimpleNamespace(
                            text=text, id=str(uuid.uuid4()), name=None
                        )
                        refdocs[file["id"]][ref["count"]] = doc
                    except:
                        pass
    jsondata_copy = copy.deepcopy(jsondata)
    if in_place:
        jsondata_out = jsondata
    else:
        jsondata_out = copy.deepcopy(jsondata)
    for file in jsondata_copy:
        for ref in file.get("refs", []):
            if file["id"] in refdocs and ref["count"] in refdocs[file["id"]]:
                doc = refdocs[file["id"]][ref["count"]]
                ref["full_text"] = {"available": True, "document_id": doc.id}
                jsondata_out.append(
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "source": "file",
                        "filename": doc.name,
                        "tag": "supporting material",
                        "url": ref["url"],
                        "refs": [],
                    }
                )
            else:
                ref["full_text"] = {"available": False}
    if not in_place:
        return jsondata_out

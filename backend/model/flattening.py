import os
import shutil
from collections import defaultdict
import hashlib
from keybert import KeyBERT

KW_MODEL = KeyBERT()

def parent(path):
    return os.path.join(path, os.pardir)

def get_files_in_directory(directory):
    """Recursively get all files in a directory."""
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files

def get_file_hash(file_path):
    """Generate a hash for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def flattenh(source, curr, target, embeddings_df):
    """List all files and directories in a given directory."""
    entries = os.listdir(curr)
    files = [os.path.join(curr, f) for f in entries if os.path.isfile(os.path.join(curr, f))]
    directories = [os.path.join(curr, d) for d in entries if os.path.isdir(os.path.join(curr, d))]
    # print("files:")
    # for f in files:
    #     print(f)
    # print("directories:")
    # for d in directories:
    #     print(d)

    if not directories:
        for f in files:
            relpath = os.path.relpath(parent(f), source)
            os.makedirs(os.path.join(target, relpath), exist_ok=True)
            shutil.copy(f, os.path.join(target, relpath))

            # Update the cluster path in the embeddings dataframe
            embeddings_df.loc[(embeddings_df["file"] == os.path.basename(f)) & (embeddings_df["cluster-path"] == relpath), "cluster-path"] = relpath

    dir_groups = defaultdict(list)
    for d in directories:
        dir_files = get_files_in_directory(d)
        file_hashset = frozenset(get_file_hash(f) for f in dir_files)
        dir_groups[file_hashset].append(d)
    

    remove = []

    for hashset, dir_paths in dir_groups.items():
        if len(dir_paths) > 1:
            names = [os.path.basename(d) for d in dir_paths]
            mergedName = "_".join(names)
            seen = set()
            for f in get_files_in_directory(dir_paths[0]):

                combined_text = "\n".join(embeddings_df.loc[embeddings_df["file"] == os.path.basename(f), "text"].to_list())
                keywords = KW_MODEL.extract_keywords(combined_text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=1)
                topic_label = "_".join([kw[0] for kw in list(set(keywords))])
                relpath = os.path.join(os.path.relpath(curr, source), topic_label)
                
                embeddings_df.loc[(embeddings_df["file"] == os.path.basename(f)) & (embeddings_df["cluster-path"] == relpath), "cluster-path"] = relpath

                if get_file_hash(f) in seen:
                    continue

                os.makedirs(os.path.join(target, relpath), exist_ok=True)
                shutil.copy(f, os.path.join(target, relpath))
                seen.add(get_file_hash(f))
            remove.append(hashset)

    for hashset in remove:
        dir_groups.pop(hashset)
    
    for _, dir_paths in dir_groups.items(): 
        for d in dir_paths:
            flattenh(source, d, target, embeddings_df)

def flatten(source, target, embeddings_df):
    flattenh(source, source, target, embeddings_df)

if __name__ == "__main__":
    import pickle
    embeddings_df = pickle.load(open(os.path.join(os.path.dirname(os.getcwd()), "data", "embeddings", "embeddings.pkl"), "rb"))
    flatten("..\\data\\organised_files", "..\\data\\target", embeddings_df)

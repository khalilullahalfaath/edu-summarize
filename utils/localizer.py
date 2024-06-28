import pandas as pd
import os


def load_bundle(locale):
    # get this directory's path
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # project root
    project_root = os.path.dirname(curr_dir)

    # bundle path
    bundle_path = os.path.join(project_root, "resources", "text_bundle.csv")

    # Load in the text bundle and filter by language locale.
    df = pd.read_csv(bundle_path)
    df = df.query(f"locale == '{locale}'")

    # Create and return a dictionary of key/values.
    lang_dict = {
        df.key.to_list()[i]: df.value.to_list()[i] for i in range(len(df.key.to_list()))
    }
    return lang_dict

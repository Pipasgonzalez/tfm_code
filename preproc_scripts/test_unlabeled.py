import lmdb
import numpy as np
import io
with lmdb.open("/scratch-shared/njagergallego/data/lvl4_unlabeled_normalized") as env:
    with env.begin() as txn:
        metadata_buffer = txn.get(b"__metadata")
        metadata_array = np.load(io.BytesIO(metadata_buffer), allow_pickle=True)
        keys, wsi_ids, coords_x, coords_y = metadata_array["keys"], metadata_array["wsi_ids"], metadata_array["coords_x"], metadata_array["coords_y"]
        np.savez(
           file = "/home/njagergallego/ablationstudy_results/unlabeled_dataset_metadata.npz",
            keys = keys,
            wsi_ids = wsi_ids,
            coords_x = coords_x,
            coords_y = coords_y
        )
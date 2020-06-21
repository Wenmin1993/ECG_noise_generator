import wfdb
import os
from IPython.display import display

def download_arrhythmia_db():
    db_dir = 'mitdb'
    dl_dir_mitdb = "E:\\Pycharm\\ECG_noise_generator\\generator\\mitdb"
    wfdb.dl_database(db_dir, dl_dir_mitdb)

if __name__ == "__main__":
    print(wfdb.get_dbs())
    download_arrhythmia_db()
    display(os.listdir(dl_dir_mitdb))

